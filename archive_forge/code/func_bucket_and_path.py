import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def bucket_and_path(self, url):
    """Split an S3-prefixed URL into bucket and path."""
    url = compat.as_str_any(url)
    if url.startswith('s3://'):
        url = url[len('s3://'):]
    idx = url.index('/')
    bucket = url[:idx]
    path = url[idx + 1:]
    return (bucket, path)