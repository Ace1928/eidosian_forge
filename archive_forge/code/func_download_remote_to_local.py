import os
from apache_beam.io.filesystems import FileSystems
from apache_beam.pipeline import Pipeline
from .logging import get_logger
def download_remote_to_local(remote_file_path, local_file_path, force_download=False):
    """Use the Beam Filesystems to download from a remote directory on gcs/s3/hdfs..."""
    fs = FileSystems
    if os.path.exists(local_file_path):
        if force_download:
            logger.info(f'Local path already exist: {remote_file_path}. Overwriting it as force_upload=True.')
        else:
            logger.info(f'Local path already exist: {remote_file_path}. Skipping it as force_upload=False.')
            return
    with fs.open(remote_file_path) as remote_file:
        with open(local_file_path, 'wb') as local_file:
            chunk = remote_file.read(CHUNK_SIZE)
            while chunk:
                local_file.write(chunk)
                chunk = remote_file.read(CHUNK_SIZE)