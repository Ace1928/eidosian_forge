import codecs
import csv
import datetime
import gettext
import glob
import os
import re
from tornado import escape
from tornado.log import gen_log
from tornado._locale_data import LOCALE_NAMES
from typing import Iterable, Any, Union, Dict, Optional
def set_default_locale(code: str) -> None:
    """Sets the default locale.

    The default locale is assumed to be the language used for all strings
    in the system. The translations loaded from disk are mappings from
    the default locale to the destination locale. Consequently, you don't
    need to create a translation file for the default locale.
    """
    global _default_locale
    global _supported_locales
    _default_locale = code
    _supported_locales = frozenset(list(_translations.keys()) + [_default_locale])