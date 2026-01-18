from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
def score_family(self, families, family2):
    """
        Return a match score between the list of font families in
        *families* and the font family name *family2*.

        An exact match at the head of the list returns 0.0.

        A match further down the list will return between 0 and 1.

        No match will return 1.0.
        """
    if not isinstance(families, (list, tuple)):
        families = [families]
    elif len(families) == 0:
        return 1.0
    family2 = family2.lower()
    step = 1 / len(families)
    for i, family1 in enumerate(families):
        family1 = family1.lower()
        if family1 in font_family_aliases:
            options = [*map(str.lower, self._expand_aliases(family1))]
            if family2 in options:
                idx = options.index(family2)
                return (i + idx / len(options)) * step
        elif family1 == family2:
            return i * step
    return 1.0