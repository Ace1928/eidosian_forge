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
def is_build_arch_any(self):
    arches = [arch for arch in self.get_architecture() if arch not in ('all', 'source')]
    return len(arches) == 1