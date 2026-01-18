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
def score_size(self, size1, size2):
    """
        Return a match score between *size1* and *size2*.

        If *size2* (the size specified in the font file) is 'scalable', this
        function always returns 0.0, since any font size can be generated.

        Otherwise, the result is the absolute distance between *size1* and
        *size2*, normalized so that the usual range of font sizes (6pt -
        72pt) will lie between 0.0 and 1.0.
        """
    if size2 == 'scalable':
        return 0.0
    try:
        sizeval1 = float(size1)
    except ValueError:
        sizeval1 = self.default_size * font_scalings[size1]
    try:
        sizeval2 = float(size2)
    except ValueError:
        return 1.0
    return abs(sizeval1 - sizeval2) / 72