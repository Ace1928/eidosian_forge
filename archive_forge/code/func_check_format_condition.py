from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def check_format_condition(condition, error_message):
    if not condition:
        raise PdfFormatError(error_message)