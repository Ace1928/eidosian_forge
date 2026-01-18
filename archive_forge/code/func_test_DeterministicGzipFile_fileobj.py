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
def test_DeterministicGzipFile_fileobj():
    with InTemporaryDirectory():
        msg = b"Hello, I'd like to have an argument."
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', fileobj=fobj, mtime=0) as gzobj:
                gzobj.write(msg)
        ref_chksum = md5sum('ref.gz')
        with open('test.gz', 'wb') as fobj:
            with DeterministicGzipFile(filename='', mode='wb', fileobj=fobj) as gzobj:
                gzobj.write(msg)
        md5sum('test.gz') == ref_chksum
        with open('test.gz', 'wb') as fobj:
            with DeterministicGzipFile(fileobj=fobj, mode='wb') as gzobj:
                gzobj.write(msg)
        md5sum('test.gz') == ref_chksum
        with open('test.gz', 'wb') as fobj:
            with DeterministicGzipFile(filename='test.gz', mode='wb', fileobj=fobj) as gzobj:
                gzobj.write(msg)
        md5sum('test.gz') == ref_chksum