import functools
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
import catalogue
import langcodes
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401
from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import OLD_MODEL_SHORTCUTS, Errors, Warnings
from .symbols import ORTH
def find_matching_language(lang: str) -> Optional[str]:
    """
    Given an IETF language code, find a supported spaCy language that is a
    close match for it (according to Unicode CLDR language-matching rules).
    This allows for language aliases, ISO 639-2 codes, more detailed language
    tags, and close matches.

    Returns the language code if a matching language is available, or None
    if there is no matching language.

    >>> find_matching_language('en')
    'en'
    >>> find_matching_language('pt-BR')  # Brazilian Portuguese
    'pt'
    >>> find_matching_language('fra')  # an ISO 639-2 code for French
    'fr'
    >>> find_matching_language('iw')  # obsolete alias for Hebrew
    'he'
    >>> find_matching_language('no')  # Norwegian
    'nb'
    >>> find_matching_language('mo')  # old code for ro-MD
    'ro'
    >>> find_matching_language('zh-Hans')  # Simplified Chinese
    'zh'
    >>> find_matching_language('zxx')
    None
    """
    import spacy.lang
    if lang == 'xx':
        return 'xx'
    possible_languages = []
    for modinfo in pkgutil.iter_modules(spacy.lang.__path__):
        code = modinfo.name
        if code == 'xx':
            possible_languages.append('mul')
        elif langcodes.tag_is_valid(code):
            possible_languages.append(code)
    match = langcodes.closest_supported_match(lang, possible_languages, max_distance=9)
    if match == 'mul':
        return 'xx'
    else:
        return match