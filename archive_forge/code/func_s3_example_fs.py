import pytest
from pyarrow.util import guid
@pytest.fixture
def s3_example_fs(s3_server):
    from pyarrow.fs import FileSystem
    host, port, access_key, secret_key = s3_server['connection']
    uri = 's3://{}:{}@mybucket/data.parquet?scheme=http&endpoint_override={}:{}'.format(access_key, secret_key, host, port)
    fs, path = FileSystem.from_uri(uri)
    fs.create_dir('mybucket')
    yield (fs, uri, path)