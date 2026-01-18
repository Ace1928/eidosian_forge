from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
@staticmethod
def logit_deformatter(string):
    """
        Parser to convert string as r'$\\mathdefault{1.41\\cdot10^{-4}}$' in
        float 1.41e-4, as '0.5' or as r'$\\mathdefault{\\frac{1}{2}}$' in float
        0.5,
        """
    match = re.match('[^\\d]*(?P<comp>1-)?(?P<mant>\\d*\\.?\\d*)?(?:\\\\cdot)?(?:10\\^\\{(?P<expo>-?\\d*)})?[^\\d]*$', string)
    if match:
        comp = match['comp'] is not None
        mantissa = float(match['mant']) if match['mant'] else 1
        expo = int(match['expo']) if match['expo'] is not None else 0
        value = mantissa * 10 ** expo
        if match['mant'] or match['expo'] is not None:
            if comp:
                return 1 - value
            return value
    match = re.match('[^\\d]*\\\\frac\\{(?P<num>\\d+)\\}\\{(?P<deno>\\d+)\\}[^\\d]*$', string)
    if match:
        num, deno = (float(match['num']), float(match['deno']))
        return num / deno
    raise ValueError('Not formatted by LogitFormatter')