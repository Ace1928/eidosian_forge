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
def test_DeterministicGzipFile():
    with InTemporaryDirectory():
        msg = b"Hello, I'd like to have an argument."
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', fileobj=fobj, mtime=0) as gzobj:
                gzobj.write(msg)
        anon_chksum = md5sum('ref.gz')
        with DeterministicGzipFile('default.gz', 'wb') as fobj:
            internal_fobj = fobj.myfileobj
            fobj.write(msg)
        assert internal_fobj.closed
        assert md5sum('default.gz') == anon_chksum
        now = time.time()
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', fileobj=fobj, mtime=now) as gzobj:
                gzobj.write(msg)
        now_chksum = md5sum('ref.gz')
        with DeterministicGzipFile('now.gz', 'wb', mtime=now) as fobj:
            fobj.write(msg)
        assert md5sum('now.gz') == now_chksum
        with mock.patch('time.time') as t:
            t.return_value = now
            with open('ref.gz', 'wb') as fobj:
                with GzipFile(filename='', mode='wb', fileobj=fobj) as gzobj:
                    gzobj.write(msg)
            assert md5sum('ref.gz') == now_chksum
            with DeterministicGzipFile('now.gz', 'wb') as fobj:
                fobj.write(msg)
            assert md5sum('now.gz') == anon_chksum
        with GzipFile('filenameA.gz', mode='wb', mtime=0) as gzobj:
            gzobj.write(msg)
        fnameA_chksum = md5sum('filenameA.gz')
        assert fnameA_chksum != anon_chksum
        with DeterministicGzipFile('filenameA.gz', 'wb') as fobj:
            fobj.write(msg)
        assert md5sum('filenameA.gz') == anon_chksum