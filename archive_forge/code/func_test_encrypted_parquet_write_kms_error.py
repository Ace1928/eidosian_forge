import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_write_kms_error(tempdir, data_table, basic_encryption_config):
    """Write an encrypted parquet, but raise KeyError in KmsClient."""
    path = tempdir / 'encrypted_table_kms_error.in_mem.parquet'
    encryption_config = basic_encryption_config
    kms_connection_config = pe.KmsConnectionConfig()

    def kms_factory(kms_connection_configuration):
        return InMemoryKmsClient(kms_connection_configuration)
    crypto_factory = pe.CryptoFactory(kms_factory)
    with pytest.raises(KeyError, match='footer_key'):
        write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)