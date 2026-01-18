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
def test_file_like_wrapper():
    message = b'History of the nude in'
    sobj = BytesIO()
    fobj = Opener(sobj)
    assert fobj.tell() == 0
    fobj.write(message)
    assert fobj.tell() == len(message)
    fobj.seek(0)
    assert fobj.tell() == 0
    assert fobj.read(6) == message[:6]
    assert not fobj.closed
    fobj.close()
    assert fobj.closed
    assert fobj.name is None