import array
import re
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from numbers import Integral
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..exceptions import NotFittedError
from ..preprocessing import normalize
from ..utils import _IS_32BIT
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils.validation import FLOAT_DTYPES, check_array, check_is_fitted
from ._hash import FeatureHasher
from ._stop_words import ENGLISH_STOP_WORDS
def strip_tags(s):
    """Basic regexp based HTML / XML tag stripper function.

    For serious HTML/XML preprocessing you should rather use an external
    library such as lxml or BeautifulSoup.

    Parameters
    ----------
    s : str
        The string to strip.

    Returns
    -------
    s : str
        The stripped string.
    """
    return re.compile('<([^>]+)>', flags=re.UNICODE).sub(' ', s)