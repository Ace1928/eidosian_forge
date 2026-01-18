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
def ls(self, path, detail=False, **kwargs):
    path = self._strip_protocol(path)
    info = self.info(path)
    if info['type'] == 'directory':
        with os.scandir(path) as it:
            infos = [self.info(f) for f in it]
    else:
        infos = [info]
    if not detail:
        return [i['name'] for i in infos]
    return infos