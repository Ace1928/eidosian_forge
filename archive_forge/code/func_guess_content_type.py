from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def guess_content_type(filename, default='application/octet-stream'):
    """
    Guess the "Content-Type" of a file.

    :param filename:
        The filename to guess the "Content-Type" of using :mod:`mimetypes`.
    :param default:
        If no "Content-Type" can be guessed, default to `default`.
    """
    if filename:
        return mimetypes.guess_type(filename)[0] or default
    return default