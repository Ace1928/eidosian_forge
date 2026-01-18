import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_cffi_mode(default=CFFI_MODE.ANY):
    cffi_mode = os.environ.get(ENVVAR_CFFI_TYPE, '')
    res = default
    for m in (CFFI_MODE.API, CFFI_MODE.ABI, CFFI_MODE.BOTH, CFFI_MODE.ANY):
        if cffi_mode.upper() == m.value:
            res = m
    logger.info(f'cffi mode is {m}')
    return res