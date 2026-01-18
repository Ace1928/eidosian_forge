import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_loop(tempdir, data_table, basic_encryption_config):
    """Write an encrypted parquet, verify it's encrypted,
    and then read it multithreaded in a loop."""
    path = tempdir / PARQUET_NAME
    encryption_config = basic_encryption_config
    kms_connection_config = pe.KmsConnectionConfig(custom_kms_conf={FOOTER_KEY_NAME: FOOTER_KEY.decode('UTF-8'), COL_KEY_NAME: COL_KEY.decode('UTF-8')})

    def kms_factory(kms_connection_configuration):
        return InMemoryKmsClient(kms_connection_configuration)
    crypto_factory = pe.CryptoFactory(kms_factory)
    write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)
    verify_file_encrypted(path)
    decryption_config = pe.DecryptionConfiguration(cache_lifetime=timedelta(minutes=5.0))
    for i in range(50):
        file_decryption_properties = crypto_factory.file_decryption_properties(kms_connection_config, decryption_config)
        assert file_decryption_properties is not None
        result = pq.ParquetFile(path, decryption_properties=file_decryption_properties)
        result_table = result.read(use_threads=True)
        assert data_table.equals(result_table)