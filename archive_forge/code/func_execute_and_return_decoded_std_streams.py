import json
import shlex
import subprocess
from typing import Tuple
import torch
def execute_and_return_decoded_std_streams(command_string):
    return _decode(subprocess.Popen(shlex.split(command_string), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate())