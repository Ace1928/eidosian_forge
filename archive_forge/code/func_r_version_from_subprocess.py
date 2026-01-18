import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def r_version_from_subprocess():
    cmd = ('R', '--version')
    logger.debug('Looking for R version with: {}'.format(' '.join(cmd)))
    try:
        tmp = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except Exception as e:
        logger.error(f'Unable to determine the R version: {e}')
        return None
    r_version = tmp.decode('ascii', 'ignore').split(os.linesep)
    if r_version[0].startswith('WARNING'):
        r_version = r_version[1]
    else:
        r_version = r_version[0].strip()
    logger.info(f'R version found: {r_version}')
    return r_version