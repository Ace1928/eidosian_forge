import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def convert_from_unicode(self, store, unicode_value):
    if self.unquote and store is not None and (unicode_value is not None):
        unicode_value = store.unquote(unicode_value)
    if self.from_unicode is None or unicode_value is None:
        return unicode_value
    try:
        converted = self.from_unicode(unicode_value)
    except (ValueError, TypeError):
        converted = None
    if converted is None and self.invalid is not None:
        if self.invalid == 'warning':
            trace.warning('Value "%s" is not valid for "%s"', unicode_value, self.name)
        elif self.invalid == 'error':
            raise ConfigOptionValueError(self.name, unicode_value)
    return converted