import codecs
import numbers
import os
import platform
import re
import subprocess
import sys
from humanfriendly.compat import coerce_string, is_unicode, on_windows, which
from humanfriendly.decorators import cached
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format
from humanfriendly.usage import format_usage
@cached
def have_windows_native_ansi_support():
    """
    Check if we're running on a Windows 10 release with native support for ANSI escape sequences.

    :returns: :data:`True` if so, :data:`False` otherwise.

    The :func:`~humanfriendly.decorators.cached` decorator is used as a minor
    performance optimization. Semantically this should have zero impact because
    the answer doesn't change in the lifetime of a computer process.
    """
    if on_windows():
        try:
            components = tuple((int(c) for c in platform.version().split('.')))
            return components >= (10, 0, 14393)
        except Exception:
            pass
    return False