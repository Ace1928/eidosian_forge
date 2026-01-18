import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@classmethod
def iter_paragraphs(cls, sequence, fields=None, use_apt_pkg=True, shared_storage=False, encoding='utf-8', strict=None):
    """Generator that yields a Deb822 object for each paragraph in Packages.

        Note that this overloaded form of the generator uses apt_pkg (a strict
        but fast parser) by default.

        See the :func:`~Deb822.iter_paragraphs` function for details.
        """
    if not strict:
        strict = {'whitespace-separates-paragraphs': False}
    return super(Packages, cls).iter_paragraphs(sequence, fields, use_apt_pkg, shared_storage, encoding, strict)