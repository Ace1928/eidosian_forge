import os
import re
import abc
import csv
import sys
import zipp
import email
import pathlib
import operator
import functools
import itertools
import posixpath
import collections
from ._compat import (
from configparser import ConfigParser
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import Any, List, Mapping, TypeVar, Union
def is_egg(self, base):
    normalized = self.legacy_normalize(self.name or '')
    prefix = normalized + '-' if normalized else ''
    versionless_egg_name = normalized + '.egg' if self.name else ''
    return base == versionless_egg_name or (base.startswith(prefix) and base.endswith('.egg'))