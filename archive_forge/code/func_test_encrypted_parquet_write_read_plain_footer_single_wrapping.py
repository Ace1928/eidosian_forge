import pytest
from datetime import timedelta
import pyarrow as pa
@pytest.mark.xfail(reason='Plaintext footer - reading plaintext column subset reads encrypted columns too')
def test_encrypted_parquet_write_read_plain_footer_single_wrapping(tempdir, data_table):
    """Write an encrypted parquet, with plaintext footer
    and with single wrapping,
    verify it's encrypted, and then read plaintext columns."""
    path = tempdir / PARQUET_NAME
    encryption_config = pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME, column_keys={COL_KEY_NAME: ['a', 'b']}, plaintext_footer=True, double_wrapping=False)
    kms_connection_config = pe.KmsConnectionConfig(custom_kms_conf={FOOTER_KEY_NAME: FOOTER_KEY.decode('UTF-8'), COL_KEY_NAME: COL_KEY.decode('UTF-8')})

    def kms_factory(kms_connection_configuration):
        return InMemoryKmsClient(kms_connection_configuration)
    crypto_factory = pe.CryptoFactory(kms_factory)
    write_encrypted_parquet(path, data_table, encryption_config, kms_connection_config, crypto_factory)