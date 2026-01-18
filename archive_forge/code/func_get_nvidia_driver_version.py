import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_nvidia_driver_version(run_lambda):
    if get_platform() == 'darwin':
        cmd = 'kextstat | grep -i cuda'
        return run_and_parse_first_match(run_lambda, cmd, 'com[.]nvidia[.]CUDA [(](.*?)[)]')
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, 'Driver Version: (.*?) ')