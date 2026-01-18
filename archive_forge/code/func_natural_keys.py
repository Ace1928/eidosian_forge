import os
import sys
import re
from collections.abc import Iterator
from warnings import warn
from looseversion import LooseVersion
import numpy as np
import textwrap
def natural_keys(text):
    if isinstance(text, tuple):
        text = text[0]
    return [atoi(c) for c in re.split('(\\d+)', text)]