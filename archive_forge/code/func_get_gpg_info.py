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
def get_gpg_info(self, keyrings=None):
    """Return a GpgInfo object with GPG signature information

        This method will raise ValueError if the signature is not available
        (e.g. the original text cannot be found).

        :param keyrings: list of keyrings to use (see GpgInfo.from_sequence)
        """
    if not hasattr(self, 'raw_text'):
        raise ValueError('original text cannot be found')
    if self.gpg_info is None:
        self.gpg_info = GpgInfo.from_sequence(self.raw_text, keyrings=keyrings)
    return self.gpg_info