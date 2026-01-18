import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_write_no_col_key(tempdir, data_table):
    """Write an encrypted parquet, but give only footer key,
    without column key."""
    path = tempdir / 'encrypted_table_no_col_key.in_mem.parquet'
    encryption_config = pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME)
    kms_connection_config = pe.KmsConnectionConfig(custom_kms_conf={FOOTER_KEY_NAME: FOOTER_KEY.decode('UTF-8'), COL_KEY_NAME: COL_KEY.decode('UTF-8')})

    def kms_factory(kms_connection_configuration):
        return InMemoryKmsClient(kms_connection_configuration)
    crypto_factory = pe.CryptoFactory(kms_factory)
    with pytest.raises(OSError, match='Either column_keys or uniform_encryption must be set'):
        write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)