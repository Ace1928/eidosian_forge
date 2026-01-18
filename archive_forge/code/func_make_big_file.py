import contextlib
import errno
import os
import resource
import sys
from breezy import osutils, tests
from breezy.tests import features, script
def make_big_file(path):
    blob_1mb = BIG_FILE_CHUNK_SIZE * b'\x0c'
    fd = os.open(path, os.O_CREAT | os.O_WRONLY)
    try:
        for i in range(BIG_FILE_SIZE // BIG_FILE_CHUNK_SIZE):
            os.write(fd, blob_1mb)
    finally:
        os.close(fd)