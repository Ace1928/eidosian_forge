import pytest
from datetime import timedelta
import pyarrow as pa
def read_encrypted_parquet(path, decryption_config, kms_connection_config, crypto_factory):
    file_decryption_properties = crypto_factory.file_decryption_properties(kms_connection_config, decryption_config)
    assert file_decryption_properties is not None
    meta = pq.read_metadata(path, decryption_properties=file_decryption_properties)
    assert meta.num_columns == 3
    schema = pq.read_schema(path, decryption_properties=file_decryption_properties)
    assert len(schema.names) == 3
    result = pq.ParquetFile(path, decryption_properties=file_decryption_properties)
    return result.read(use_threads=True)