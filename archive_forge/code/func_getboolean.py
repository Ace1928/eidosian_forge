from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def getboolean(self, section, option, *, raw=False, vars=None, fallback=_UNSET, **kwargs):
    return self._get_conv(section, option, self._convert_to_boolean, raw=raw, vars=vars, fallback=fallback, **kwargs)