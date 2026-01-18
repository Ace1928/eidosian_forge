import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def test_no_retry_on_success(self):
    fd, tmpfilename = tempfile.mkstemp()
    _, tmpfilename2 = tempfile.mkstemp()
    try:
        fp = os.fdopen(fd, 'w+')
        fp.write('#!/bin/sh\n# If we\'ve already run, bail out.\ngrep -q foo "$1" && exit 1\n# Mark that we\'ve run before.\necho foo > "$1"\n# Check that stdin gets passed correctly.\ngrep foo\n')
        fp.close()
        os.chmod(tmpfilename, 493)
        processutils.execute(tmpfilename, tmpfilename2, process_input=b'foo', attempts=2)
    finally:
        os.unlink(tmpfilename)
        os.unlink(tmpfilename2)