import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_r_flags(r_home: str, flags: str):
    """Get the parsed output of calling 'R CMD CONFIG <about>'.

    Returns a tuple (parsed_args, unknown_args), with parsed_args
    having the attribute `l`, 'L', and 'I'."""
    assert flags in _R_FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', action='append')
    parser.add_argument('-L', action='append')
    parser.add_argument('-l', action='append')
    res = shlex.split(' '.join(_get_r_cmd_config(r_home, flags, allow_empty=False)))
    return parser.parse_known_args(res)