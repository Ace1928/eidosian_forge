from datetime import timedelta
import pyarrow.fs as fs
import pyarrow as pa
import pytest
def create_decryption_config():
    return pe.DecryptionConfiguration(cache_lifetime=300)