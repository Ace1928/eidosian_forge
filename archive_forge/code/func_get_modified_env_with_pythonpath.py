import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def get_modified_env_with_pythonpath():
    env = os.environ.copy()
    existing_pythonpath = env.get('PYTHONPATH', '')
    module_path = os.path.abspath(os.path.dirname(os.path.dirname(pa.__file__)))
    if existing_pythonpath:
        new_pythonpath = os.pathsep.join((module_path, existing_pythonpath))
    else:
        new_pythonpath = module_path
    env['PYTHONPATH'] = new_pythonpath
    return env