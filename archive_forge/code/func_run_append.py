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
def run_append(self, args):
    sztarget: str = args.arcfile
    filenames: List[str] = args.filenames
    if not sztarget.endswith('.7z'):
        sys.stderr.write('Error: specified archive file is invalid.')
        self.show_help(args)
        exit(1)
    target = pathlib.Path(sztarget)
    if not target.exists():
        sys.stderr.write('Archive file does not exists!\n')
        self.show_help(args)
        exit(1)
    with py7zr.SevenZipFile(target, 'a') as szf:
        for path in filenames:
            src = pathlib.Path(path)
            if src.is_dir():
                szf.writeall(src)
            else:
                szf.write(src)
    return 0