import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
def set_output_redirected(val: bool):
    """Choose between logging to a temporary file and to `sys.stdout`.

    The default is to log to a file.

    Args:
        val: If true, subprocess output will be redirected to
                    a temporary file.
    """
    global _redirect_output
    _redirect_output = val