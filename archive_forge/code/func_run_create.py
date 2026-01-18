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
def run_create(self, args):
    sztarget = args.arcfile
    filenames = args.filenames
    volume_size = args.volume[0] if getattr(args, 'volume', None) is not None else None
    if volume_size is not None and (not self._check_volumesize_valid(volume_size)):
        sys.stderr.write('Error: Specified volume size is invalid.\n')
        self.show_help(args)
        exit(1)
    if not sztarget.endswith('.7z'):
        sztarget += '.7z'
    target = pathlib.Path(sztarget)
    if target.exists():
        sys.stderr.write('Archive file exists!\n')
        self.show_help(args)
        exit(1)
    if not args.password:
        password = None
    else:
        try:
            password = getpass.getpass()
        except getpass.GetPassWarning:
            sys.stderr.write('Warning: your password may be shown.\n')
            return 1
    if volume_size is None:
        with py7zr.SevenZipFile(target, 'w', password=password) as szf:
            for path in filenames:
                src = pathlib.Path(path)
                if src.is_dir():
                    szf.writeall(src)
                else:
                    szf.write(src)
        return 0
    else:
        size = self._volumesize_unitconv(volume_size)
        with multivolumefile.MultiVolume(target, mode='wb', volume=size, ext_digits=4) as mvf:
            with py7zr.SevenZipFile(mvf, 'w', password=password) as szf:
                for path in filenames:
                    src = pathlib.Path(path)
                    if src.is_dir():
                        szf.writeall(src)
                    else:
                        szf.write(src)
        return 0