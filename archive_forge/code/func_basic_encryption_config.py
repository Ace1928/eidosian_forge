import pytest
from datetime import timedelta
import pyarrow as pa
@pytest.fixture(scope='module')
def basic_encryption_config():
    basic_encryption_config = pe.EncryptionConfiguration(footer_key=FOOTER_KEY_NAME, column_keys={COL_KEY_NAME: ['a', 'b']})
    return basic_encryption_config