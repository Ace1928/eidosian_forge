from __future__ import unicode_literals
import base64
import codecs
import datetime
from email import message_from_file
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import sys
import tempfile
import zipfile
from . import __version__, DistlibException
from .compat import sysconfig, ZipFile, fsdecode, text_type, filter
from .database import InstalledDistribution
from .metadata import Metadata, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME
from .util import (FileOperator, convert_path, CSVReader, CSVWriter, Cache,
from .version import NormalizedVersion, UnsupportedVersionError
def build_zip(self, pathname, archive_paths):
    with ZipFile(pathname, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ap, p in archive_paths:
            logger.debug('Wrote %s to %s in wheel', p, ap)
            zf.write(p, ap)