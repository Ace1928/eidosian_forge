import datetime
import importlib.metadata
import os
import posixpath
import urllib.parse
from collections import namedtuple
from packaging.version import Version
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import (
from mlflow.exceptions import _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repo import ArtifactRepository, MultipartUploadMixin
from mlflow.utils.file_utils import relative_path_to_artifact_path
def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
    self._validate_support_mpu()
    from google.resumable_media.requests import XMLMPUContainer
    bucket, dest_path = self.parse_gcs_uri(self.artifact_uri)
    if artifact_path:
        dest_path = posixpath.join(dest_path, artifact_path)
    dest_path = posixpath.join(dest_path, os.path.basename(local_file))
    gcs_bucket = self._get_bucket(bucket)
    blob = gcs_bucket.blob(dest_path)
    args = self._gcs_mpu_arguments(local_file, blob)
    container = XMLMPUContainer(args.url, local_file, headers=args.headers)
    container.initiate(transport=args.transport, content_type=args.content_type)
    upload_id = container.upload_id
    credentials = []
    for i in range(1, num_parts + 1):
        signed_url = blob.generate_signed_url(method='PUT', version='v4', expiration=datetime.timedelta(minutes=60), query_parameters={'partNumber': i, 'uploadId': upload_id})
        credentials.append(MultipartUploadCredential(url=signed_url, part_number=i, headers={}))
    return CreateMultipartUploadResponse(credentials=credentials, upload_id=upload_id)