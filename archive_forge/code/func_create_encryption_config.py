from datetime import timedelta
import pyarrow.fs as fs
import pyarrow as pa
import pytest
def create_encryption_config():
    return pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME, plaintext_footer=False, column_keys={COL_KEY_NAME: ['n_legs', 'animal']}, encryption_algorithm='AES_GCM_V1', cache_lifetime=timedelta(minutes=5.0), data_key_length_bits=256)