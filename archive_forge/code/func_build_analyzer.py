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
def build_analyzer(self):
    """Return a callable to process input data.

        The callable handles preprocessing, tokenization, and n-grams generation.

        Returns
        -------
        analyzer: callable
            A function to handle preprocessing, tokenization
            and n-grams generation.
        """
    if callable(self.analyzer):
        return partial(_analyze, analyzer=self.analyzer, decoder=self.decode)
    preprocess = self.build_preprocessor()
    if self.analyzer == 'char':
        return partial(_analyze, ngrams=self._char_ngrams, preprocessor=preprocess, decoder=self.decode)
    elif self.analyzer == 'char_wb':
        return partial(_analyze, ngrams=self._char_wb_ngrams, preprocessor=preprocess, decoder=self.decode)
    elif self.analyzer == 'word':
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        self._check_stop_words_consistency(stop_words, preprocess, tokenize)
        return partial(_analyze, ngrams=self._word_ngrams, tokenizer=tokenize, preprocessor=preprocess, decoder=self.decode, stop_words=stop_words)
    else:
        raise ValueError('%s is not a valid tokenization scheme/analyzer' % self.analyzer)