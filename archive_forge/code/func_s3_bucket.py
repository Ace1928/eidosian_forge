import pytest
from pyarrow.util import guid
@pytest.fixture
def s3_bucket(s3_server):
    boto3 = pytest.importorskip('boto3')
    botocore = pytest.importorskip('botocore')
    s3_bucket_name = 'test-s3fs'
    host, port, access_key, secret_key = s3_server['connection']
    s3_client = boto3.client('s3', endpoint_url='http://{}:{}'.format(host, port), aws_access_key_id=access_key, aws_secret_access_key=secret_key, config=botocore.client.Config(signature_version='s3v4'), region_name='us-east-1')
    try:
        s3_client.create_bucket(Bucket=s3_bucket_name)
    except Exception:
        pass
    finally:
        s3_client.close()
    return s3_bucket_name