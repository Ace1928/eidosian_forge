import os
import platform
import subprocess
import sys
from shutil import which
from typing import List
import torch
def parse_choice_from_env(key, default='no'):
    value = os.environ.get(key, str(default))
    return value