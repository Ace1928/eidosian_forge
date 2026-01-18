import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
def get_zip_archive(self, url, dirOffset=0):
    if self._fname is None:
        raise DeveloperError('target file name has not been initialized with set_destination_filename')
    if os.path.exists(self._fname) and (not os.path.isdir(self._fname)):
        raise RuntimeError('Target directory (%s) exists, but is not a directory' % (self._fname,))
    zip_file = zipfile.ZipFile(io.BytesIO(self.retrieve_url(url)))
    for info in zip_file.infolist():
        f = info.filename
        if f[0] in '\\/' or '..' in f:
            logger.error('malformed (potentially insecure) filename (%s) found in zip archive.  Skipping file.' % (f,))
            continue
        target = self._splitpath(f)
        if len(target) <= dirOffset:
            if f[-1] != '/':
                logger.warning('Skipping file (%s) in zip archive due to dirOffset' % (f,))
            continue
        info.filename = target[-1] + '/' if f[-1] == '/' else target[-1]
        zip_file.extract(f, os.path.join(self._fname, *tuple(target[dirOffset:-1])))