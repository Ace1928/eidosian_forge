import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def align_center(text: str, *, fill_char: str=' ', width: Optional[int]=None, tab_width: int=4, truncate: bool=False) -> str:
    """
    Center text for display within a given width. Supports characters with display widths greater than 1.
    ANSI style sequences do not count toward the display width. If text has line breaks, then each line is aligned
    independently.

    :param text: text to center (can contain multiple lines)
    :param fill_char: character that fills the alignment gap. Defaults to space. (Cannot be a line breaking character)
    :param width: display width of the aligned text. Defaults to width of the terminal.
    :param tab_width: any tabs in the text will be replaced with this many spaces. if fill_char is a tab, then it will
                      be converted to one space.
    :param truncate: if True, then text will be shortened to fit within the display width. The truncated portion is
                     replaced by a '…' character. Defaults to False.
    :return: centered text
    :raises: TypeError if fill_char is more than one character (not including ANSI style sequences)
    :raises: ValueError if text or fill_char contains an unprintable character
    :raises: ValueError if width is less than 1
    """
    return align_text(text, TextAlignment.CENTER, fill_char=fill_char, width=width, tab_width=tab_width, truncate=truncate)