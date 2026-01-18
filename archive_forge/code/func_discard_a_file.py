import datetime
import logging
import os
import types
import uuid
from stat import S_ISDIR, S_ISLNK
import paramiko
from .. import AbstractFileSystem
from ..utils import infer_storage_options
def discard_a_file(self):
    self.fs._rm(self.temppath)