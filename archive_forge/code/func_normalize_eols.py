import glob
import os
import re
import sys
from functools import total_ordering
from itertools import dropwhile
from pathlib import Path
import django
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.files.temp import NamedTemporaryFile
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import (
from django.utils.encoding import DEFAULT_LOCALE_ENCODING
from django.utils.functional import cached_property
from django.utils.jslex import prepare_js_for_gettext
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import get_text_list
from django.utils.translation import templatize
def normalize_eols(raw_contents):
    """
    Take a block of raw text that will be passed through str.splitlines() to
    get universal newlines treatment.

    Return the resulting block of text with normalized `
` EOL sequences ready
    to be written to disk using current platform's native EOLs.
    """
    lines_list = raw_contents.splitlines()
    if lines_list and lines_list[-1]:
        lines_list.append('')
    return '\n'.join(lines_list)