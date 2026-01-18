import functools
import glob
import gzip
import os
import sys
import warnings
import zipfile
from itertools import product
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.core.management.utils import parse_apps_and_model_labels
from django.db import (
from django.utils.functional import cached_property
@cached_property
def compression_formats(self):
    """A dict mapping format names to (open function, mode arg) tuples."""
    compression_formats = {None: (open, 'rb'), 'gz': (gzip.GzipFile, 'rb'), 'zip': (SingleZipReader, 'r'), 'stdin': (lambda *args: sys.stdin, None)}
    if has_bz2:
        compression_formats['bz2'] = (bz2.BZ2File, 'r')
    if has_lzma:
        compression_formats['lzma'] = (lzma.LZMAFile, 'r')
        compression_formats['xz'] = (lzma.LZMAFile, 'r')
    return compression_formats