from __future__ import annotations
import datetime
import fnmatch
import logging
import optparse
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from configparser import RawConfigParser
from io import StringIO
from typing import Iterable
from babel import Locale, localedata
from babel import __version__ as VERSION
from babel.core import UnknownLocaleError
from babel.messages.catalog import DEFAULT_HEADER, Catalog
from babel.messages.extract import (
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po, write_po
from babel.util import LOCALTZ
def parse_mapping(fileobj, filename=None):
    """Parse an extraction method mapping from a file-like object.

    >>> buf = StringIO('''
    ... [extractors]
    ... custom = mypackage.module:myfunc
    ...
    ... # Python source files
    ... [python: **.py]
    ...
    ... # Genshi templates
    ... [genshi: **/templates/**.html]
    ... include_attrs =
    ... [genshi: **/templates/**.txt]
    ... template_class = genshi.template:TextTemplate
    ... encoding = latin-1
    ...
    ... # Some custom extractor
    ... [custom: **/custom/*.*]
    ... ''')

    >>> method_map, options_map = parse_mapping(buf)
    >>> len(method_map)
    4

    >>> method_map[0]
    ('**.py', 'python')
    >>> options_map['**.py']
    {}
    >>> method_map[1]
    ('**/templates/**.html', 'genshi')
    >>> options_map['**/templates/**.html']['include_attrs']
    ''
    >>> method_map[2]
    ('**/templates/**.txt', 'genshi')
    >>> options_map['**/templates/**.txt']['template_class']
    'genshi.template:TextTemplate'
    >>> options_map['**/templates/**.txt']['encoding']
    'latin-1'

    >>> method_map[3]
    ('**/custom/*.*', 'mypackage.module:myfunc')
    >>> options_map['**/custom/*.*']
    {}

    :param fileobj: a readable file-like object containing the configuration
                    text to parse
    :see: `extract_from_directory`
    """
    extractors = {}
    method_map = []
    options_map = {}
    parser = RawConfigParser()
    parser._sections = OrderedDict(parser._sections)
    parser.read_file(fileobj, filename)
    for section in parser.sections():
        if section == 'extractors':
            extractors = dict(parser.items(section))
        else:
            method, pattern = (part.strip() for part in section.split(':', 1))
            method_map.append((pattern, method))
            options_map[pattern] = dict(parser.items(section))
    if extractors:
        for idx, (pattern, method) in enumerate(method_map):
            if method in extractors:
                method = extractors[method]
            method_map[idx] = (pattern, method)
    return (method_map, options_map)