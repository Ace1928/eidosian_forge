import codecs
import logging
import os
import pickle
import re
import shutil
import tarfile
from contextlib import suppress
import joblib
import numpy as np
import scipy.sparse as sp
from .. import preprocessing
from ..feature_extraction.text import CountVectorizer
from ..utils import Bunch, check_random_state
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home, load_files
from ._base import (
def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break
    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text