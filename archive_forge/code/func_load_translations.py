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
def load_translations(directory: str, encoding: Optional[str]=None) -> None:
    """Loads translations from CSV files in a directory.

    Translations are strings with optional Python-style named placeholders
    (e.g., ``My name is %(name)s``) and their associated translations.

    The directory should have translation files of the form ``LOCALE.csv``,
    e.g. ``es_GT.csv``. The CSV files should have two or three columns: string,
    translation, and an optional plural indicator. Plural indicators should
    be one of "plural" or "singular". A given string can have both singular
    and plural forms. For example ``%(name)s liked this`` may have a
    different verb conjugation depending on whether %(name)s is one
    name or a list of names. There should be two rows in the CSV file for
    that string, one with plural indicator "singular", and one "plural".
    For strings with no verbs that would change on translation, simply
    use "unknown" or the empty string (or don't include the column at all).

    The file is read using the `csv` module in the default "excel" dialect.
    In this format there should not be spaces after the commas.

    If no ``encoding`` parameter is given, the encoding will be
    detected automatically (among UTF-8 and UTF-16) if the file
    contains a byte-order marker (BOM), defaulting to UTF-8 if no BOM
    is present.

    Example translation ``es_LA.csv``::

        "I love you","Te amo"
        "%(name)s liked this","A %(name)s les gustó esto","plural"
        "%(name)s liked this","A %(name)s le gustó esto","singular"

    .. versionchanged:: 4.3
       Added ``encoding`` parameter. Added support for BOM-based encoding
       detection, UTF-16, and UTF-8-with-BOM.
    """
    global _translations
    global _supported_locales
    _translations = {}
    for path in os.listdir(directory):
        if not path.endswith('.csv'):
            continue
        locale, extension = path.split('.')
        if not re.match('[a-z]+(_[A-Z]+)?$', locale):
            gen_log.error('Unrecognized locale %r (path: %s)', locale, os.path.join(directory, path))
            continue
        full_path = os.path.join(directory, path)
        if encoding is None:
            with open(full_path, 'rb') as bf:
                data = bf.read(len(codecs.BOM_UTF16_LE))
            if data in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
                encoding = 'utf-16'
            else:
                encoding = 'utf-8-sig'
        with open(full_path, encoding=encoding) as f:
            _translations[locale] = {}
            for i, row in enumerate(csv.reader(f)):
                if not row or len(row) < 2:
                    continue
                row = [escape.to_unicode(c).strip() for c in row]
                english, translation = row[:2]
                if len(row) > 2:
                    plural = row[2] or 'unknown'
                else:
                    plural = 'unknown'
                if plural not in ('plural', 'singular', 'unknown'):
                    gen_log.error('Unrecognized plural indicator %r in %s line %d', plural, path, i + 1)
                    continue
                _translations[locale].setdefault(plural, {})[english] = translation
    _supported_locales = frozenset(list(_translations.keys()) + [_default_locale])
    gen_log.debug('Supported locales: %s', sorted(_supported_locales))