import contextlib
import hashlib
import os
import time
import unittest
from gzip import GzipFile
from io import BytesIO, UnsupportedOperation
from unittest import mock
import pytest
from packaging.version import Version
from ..deprecator import ExpiredDeprecationError
from ..openers import HAVE_INDEXED_GZIP, BZ2File, DeterministicGzipFile, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
def test_compressionlevel():
    with open(__file__, 'rb') as fobj:
        my_self = fobj.read()
    many_selves = my_self * 50

    class MyOpener(Opener):
        default_compresslevel = 5
    with InTemporaryDirectory():
        for ext in ('gz', 'bz2', 'GZ', 'gZ', 'BZ2', 'Bz2'):
            for opener, default_val in ((Opener, 1), (MyOpener, 5)):
                sizes = {}
                for compresslevel in ('default', 1, 5):
                    fname = 'test.' + ext
                    kwargs = {'mode': 'wb'}
                    if compresslevel != 'default':
                        kwargs['compresslevel'] = compresslevel
                    with opener(fname, **kwargs) as fobj:
                        fobj.write(many_selves)
                    with open(fname, 'rb') as fobj:
                        my_selves_smaller = fobj.read()
                    sizes[compresslevel] = len(my_selves_smaller)
                assert sizes['default'] == sizes[default_val]
                assert sizes[1] > sizes[5]