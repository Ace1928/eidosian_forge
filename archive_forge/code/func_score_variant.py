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
def score_variant(self, variant1, variant2):
    """
        Return a match score between *variant1* and *variant2*.

        An exact match returns 0.0, otherwise 1.0.
        """
    if variant1 == variant2:
        return 0.0
    else:
        return 1.0