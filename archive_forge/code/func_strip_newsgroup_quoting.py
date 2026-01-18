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
def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    good_lines = [line for line in text.split('\n') if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)