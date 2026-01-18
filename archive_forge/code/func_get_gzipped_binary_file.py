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
def get_gzipped_binary_file(self, url):
    if self._fname is None:
        raise DeveloperError('target file name has not been initialized with set_destination_filename')
    with open(self._fname, 'wb') as FILE:
        gzipped_file = io.BytesIO(self.retrieve_url(url))
        raw_file = gzip.GzipFile(fileobj=gzipped_file).read()
        FILE.write(raw_file)
        logger.info('  ...wrote %s bytes' % (len(raw_file),))