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
def run_extract(self, args: argparse.Namespace) -> int:
    target = args.arcfile
    verbose = args.verbose
    if not py7zr.is_7zfile(target):
        print('not a 7z file')
        return 1
    if not args.password:
        password = None
    else:
        try:
            password = getpass.getpass()
        except getpass.GetPassWarning:
            sys.stderr.write('Warning: your password may be shown.\n')
            return 1
    try:
        a = py7zr.SevenZipFile(target, 'r', password=password)
    except py7zr.exceptions.Bad7zFile:
        print('Header is corrupted. Cannot read as 7z file.')
        return 1
    except py7zr.exceptions.PasswordRequired:
        print('The archive is encrypted, but password is not given. ABORT.')
        return 1
    except lzma.LZMAError:
        if password is None:
            print('The archive is corrupted. ABORT.')
        else:
            print('The archive is corrupted, or password is wrong. ABORT.')
        return 1
    except _lzma.LZMAError:
        return 1
    cb = None
    if verbose:
        archive_info = a.archiveinfo()
        cb = CliExtractCallback(total_bytes=archive_info.uncompressed, ofd=sys.stderr)
    try:
        if args.odir:
            a.extractall(path=args.odir, callback=cb)
        else:
            a.extractall(callback=cb)
    except py7zr.exceptions.UnsupportedCompressionMethodError:
        print('Unsupported compression method is used in archive. ABORT.')
        return 1
    except py7zr.exceptions.DecompressionError:
        print('Error has been occurred during decompression. ABORT.')
        return 1
    except py7zr.exceptions.PasswordRequired:
        print('The archive is encrypted, but password is not given. ABORT.')
        return 1
    except lzma.LZMAError:
        if password is None:
            print('The archive is corrupted. ABORT.')
        else:
            print('The archive is corrupted, or password is wrong. ABORT.')
        return 1
    except _lzma.LZMAError:
        return 1
    else:
        return 0