import os
import unittest
import contextlib
import tempfile
import shutil
import io
import signal
from typing import Tuple, Dict, Any
from parlai.core.opt import Opt
import parlai.utils.logging as logging
def is_new_task_filename(filename):
    """
    Check if a given filename counts as a new task.

    Used in tests and test triggers, and only here to avoid redundancy.
    """
    return 'parlai/tasks' in filename and 'README' not in filename and ('task_list.py' not in filename)