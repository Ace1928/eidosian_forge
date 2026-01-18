from __future__ import annotations
import collections.abc
import copy
import functools
import itertools
import operator
import random
import re
from collections.abc import Container, Iterable, Mapping
from typing import Any, Callable, Union
import jaraco.text
def pairkey_key(item):
    return key(item[0])