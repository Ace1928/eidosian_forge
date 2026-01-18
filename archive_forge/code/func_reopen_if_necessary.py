import argparse
import errno
import glob
import logging
import logging.handlers
import os
import platform
import re
import shutil
import time
import traceback
from typing import Callable, List, Optional, Set
from ray._raylet import GcsClient
import ray._private.ray_constants as ray_constants
import ray._private.services as services
import ray._private.utils
from ray._private.ray_logging import setup_component_logger
def reopen_if_necessary(self):
    """Check if the file's inode has changed and reopen it if necessary.
        There are a variety of reasons what we would logically consider a file
        would have different inodes, such as log rotation or file syncing
        semantics.
        """
    try:
        open_inode = None
        if self.file_handle and (not self.file_handle.closed):
            open_inode = os.fstat(self.file_handle.fileno()).st_ino
        new_inode = os.stat(self.filename).st_ino
        if open_inode != new_inode:
            self.file_handle = open(self.filename, 'rb')
            self.file_handle.seek(self.file_position)
    except Exception:
        logger.debug(f'file no longer exists, skip re-opening of {self.filename}')