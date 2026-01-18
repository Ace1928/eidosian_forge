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
def set_size_field_behavior(self, value):
    if value not in ['apt-ftparchive', 'dak']:
        raise ValueError("size_field_behavior must be either 'apt-ftparchive' or 'dak'")
    self.__size_field_behavior = value