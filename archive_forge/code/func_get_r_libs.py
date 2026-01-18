import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_r_libs(r_home: str, libs: str):
    assert libs in _R_LIBS
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', action='append')
    parser.add_argument('-L', action='append')
    parser.add_argument('-l', action='append')
    res = shlex.split(' '.join(_get_r_cmd_config(r_home, libs, allow_empty=False)))
    return parser.parse_known_args(res)