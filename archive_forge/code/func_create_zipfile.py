from base64 import standard_b64encode
from distutils import log
from distutils.errors import DistutilsOptionError
import os
import zipfile
import tempfile
import shutil
import itertools
import functools
import http.client
import urllib.parse
from .._importlib import metadata
from ..warnings import SetuptoolsDeprecationWarning
from .upload import upload
def create_zipfile(self, filename):
    zip_file = zipfile.ZipFile(filename, 'w')
    try:
        self.mkpath(self.target_dir)
        for root, dirs, files in os.walk(self.target_dir):
            if root == self.target_dir and (not files):
                tmpl = "no files found in upload directory '%s'"
                raise DistutilsOptionError(tmpl % self.target_dir)
            for name in files:
                full = os.path.join(root, name)
                relative = root[len(self.target_dir):].lstrip(os.path.sep)
                dest = os.path.join(relative, name)
                zip_file.write(full, dest)
    finally:
        zip_file.close()