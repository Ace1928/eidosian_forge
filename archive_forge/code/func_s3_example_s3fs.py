import pytest
from pyarrow.util import guid
@pytest.fixture
def s3_example_s3fs(s3_server, s3_bucket):
    s3fs = pytest.importorskip('s3fs')
    host, port, access_key, secret_key = s3_server['connection']
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, client_kwargs={'endpoint_url': 'http://{}:{}'.format(host, port)})
    test_path = '{}/{}'.format(s3_bucket, guid())
    fs.mkdir(test_path)
    yield (fs, test_path)
    try:
        fs.rm(test_path, recursive=True)
    except FileNotFoundError:
        pass