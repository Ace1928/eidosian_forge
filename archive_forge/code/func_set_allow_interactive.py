import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
def set_allow_interactive(val: bool):
    """Choose whether to pass on stdin to running commands.

    The default is to pipe stdin and close it immediately.

    Args:
        val: If true, stdin will be passed to commands.
    """
    global _allow_interactive
    _allow_interactive = val