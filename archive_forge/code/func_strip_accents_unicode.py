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
def strip_accents_unicode(s):
    """Transform accentuated unicode symbols into their simple counterpart.

    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.

    Parameters
    ----------
    s : str
        The string to strip.

    Returns
    -------
    s : str
        The stripped string.

    See Also
    --------
    strip_accents_ascii : Remove accentuated char for any unicode symbol that
        has a direct ASCII equivalent.
    """
    try:
        s.encode('ASCII', errors='strict')
        return s
    except UnicodeEncodeError:
        normalized = unicodedata.normalize('NFKD', s)
        return ''.join([c for c in normalized if not unicodedata.combining(c)])