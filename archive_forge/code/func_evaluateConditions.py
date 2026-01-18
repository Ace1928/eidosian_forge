from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
def evaluateConditions(conditions, location):
    """Return True if all the conditions matches the given location.

    - If a condition has no minimum, check for < maximum.
    - If a condition has no maximum, check for > minimum.
    """
    for cd in conditions:
        value = location[cd['name']]
        if cd.get('minimum') is None:
            if value > cd['maximum']:
                return False
        elif cd.get('maximum') is None:
            if cd['minimum'] > value:
                return False
        elif not cd['minimum'] <= value <= cd['maximum']:
            return False
    return True