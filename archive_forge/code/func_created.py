import datetime
import io
import logging
import os
import os.path as osp
import re
import shutil
import stat
import tempfile
from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.utils import isfilelike, stringify_path
def created(self, path):
    info = self.info(path=path)
    return datetime.datetime.fromtimestamp(info['created'], tz=datetime.timezone.utc)