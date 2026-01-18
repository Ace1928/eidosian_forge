import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def put_file_non_atomic(self, relpath, f, mode=None, create_parent_dir=False, dir_mode=False):
    self.put_bytes_non_atomic(relpath, f.read(), mode=mode, create_parent_dir=create_parent_dir, dir_mode=dir_mode)