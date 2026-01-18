from urllib.parse import urlparse
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client
@staticmethod
def convert_r2_uri_to_s3_endpoint_url(r2_uri):
    host = urlparse(r2_uri).netloc
    host_without_bucket = host.split('@')[-1]
    return f'https://{host_without_bucket}'