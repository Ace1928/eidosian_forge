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
@staticmethod
def split_gpg_and_payload(sequence, strict=None):
    """Return a (gpg_pre, payload, gpg_post) tuple

        Each element of the returned tuple is a list of lines (with trailing
        whitespace stripped).

        :param sequence: iterable.
            An iterable that yields lines of data (str, unicode,
            bytes) to be parsed, possibly including a GPG in-line signature.
        :param strict: dict, optional.
            Control over the strictness of the parser. See the :class:`Deb822`
            class documentation for details.
        """
    _encoded_sequence = (x.encode() if isinstance(x, str) else x for x in sequence)
    return Deb822._split_gpg_and_payload(_encoded_sequence, strict=strict)