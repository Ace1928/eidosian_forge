import base64
import dataclasses
import datetime
import errno
import json
import os
import subprocess
import tempfile
import time
import typing
from typing import Optional
from tensorboard import version
from tensorboard.util import tb_logging
def remove_info_file():
    """Remove the current process's TensorBoardInfo file, if it exists.

    If the file does not exist, no action is taken and no error is
    raised.
    """
    try:
        os.unlink(_get_info_file_path())
    except OSError as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise