import argparse
import getpass
import inspect
import io
import lzma
import os
import pathlib
import platform
import re
import shutil
import sys
from lzma import CHECK_CRC64, CHECK_SHA256, is_check_supported
from typing import Any, List, Optional
import _lzma  # type: ignore
import multivolumefile
import texttable  # type: ignore
import py7zr
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods
from py7zr.helpers import Local
from py7zr.properties import COMMAND_HELP_STRING
def run_list(self, args):
    """Print a table of contents to file."""
    target = args.arcfile
    verbose = args.verbose
    if re.fullmatch('[.]0+1?', target.suffix):
        mv_target = pathlib.Path(target.parent, target.stem)
        ext_start = int(target.suffix[-1])
        with multivolumefile.MultiVolume(mv_target, mode='rb', ext_digits=len(target.suffix) - 1, ext_start=ext_start) as mvf:
            setattr(mvf, 'name', str(mv_target))
            return self._run_list(mvf, verbose)
    else:
        return self._run_list(target, verbose)