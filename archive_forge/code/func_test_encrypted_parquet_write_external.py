import pytest
from datetime import timedelta
import pyarrow as pa
@pytest.mark.xfail(reason='External key material not supported yet')
def test_encrypted_parquet_write_external(tempdir, data_table):
    """Write an encrypted parquet, with external key
    material.
    Currently it's not implemented, so should throw
    an exception"""
    path = tempdir / PARQUET_NAME
    encryption_config = pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME, column_keys={}, internal_key_material=False)
    kms_connection_config = pe.KmsConnectionConfig(custom_kms_conf={FOOTER_KEY_NAME: FOOTER_KEY.decode('UTF-8')})

    def kms_factory(kms_connection_configuration):
        return InMemoryKmsClient(kms_connection_configuration)
    crypto_factory = pe.CryptoFactory(kms_factory)
    write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)