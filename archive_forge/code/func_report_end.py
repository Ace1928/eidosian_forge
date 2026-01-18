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
def report_end(self, processing_file_path, wrote_bytes):
    self.total_bytes += int(wrote_bytes)
    plest = self.columns - self.pwidth
    progress = self.total_bytes / self.archive_total
    msg = '({:.0%})\n'.format(progress)
    if plest - len(msg) > 0:
        self.ofd.write(msg.rjust(plest))
    else:
        self.ofd.write(msg)
    self.pwidth = 0