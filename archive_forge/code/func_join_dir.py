import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def join_dir(base_path, path):
    """
    Returns an absolute path to C{path}, calculated relative to the parent
    of C{base_path}.

    @param base_path: A path to a file or directory.
    @param path: An absolute path, or a path relative to the containing
    directory of C{base_path}.

    @return: An absolute path to C{path}.
    """
    return os.path.join(os.path.dirname(os.path.abspath(base_path)), path)