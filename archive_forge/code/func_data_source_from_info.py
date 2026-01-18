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
def data_source_from_info(info):
    """Format the data location for the given TensorBoardInfo.

    Args:
      info: A TensorBoardInfo value.

    Returns:
      A human-readable string describing the logdir or database connection
      used by the server: e.g., "logdir /tmp/logs".
    """
    if info.db:
        return 'db %s' % info.db
    else:
        return 'logdir %s' % info.logdir