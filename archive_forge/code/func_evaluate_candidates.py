import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..exceptions import NotFittedError
from ..metrics import check_scoring
from ..metrics._scorer import (
from ..utils import Bunch, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_method_params, check_is_fitted, indexable
from ._split import check_cv
from ._validation import (
def evaluate_candidates(candidate_params, cv=None, more_results=None):
    cv = cv or cv_orig
    candidate_params = list(candidate_params)
    n_candidates = len(candidate_params)
    if self.verbose > 0:
        print('Fitting {0} folds for each of {1} candidates, totalling {2} fits'.format(n_splits, n_candidates, n_candidates * n_splits))
    out = parallel((delayed(_fit_and_score)(clone(base_estimator), X, y, train=train, test=test, parameters=parameters, split_progress=(split_idx, n_splits), candidate_progress=(cand_idx, n_candidates), **fit_and_score_kwargs) for (cand_idx, parameters), (split_idx, (train, test)) in product(enumerate(candidate_params), enumerate(cv.split(X, y, **routed_params.splitter.split)))))
    if len(out) < 1:
        raise ValueError('No fits were performed. Was the CV iterator empty? Were there no candidates?')
    elif len(out) != n_candidates * n_splits:
        raise ValueError('cv.split and cv.get_n_splits returned inconsistent results. Expected {} splits, got {}'.format(n_splits, len(out) // n_candidates))
    _warn_or_raise_about_fit_failures(out, self.error_score)
    if callable(self.scoring):
        _insert_error_scores(out, self.error_score)
    all_candidate_params.extend(candidate_params)
    all_out.extend(out)
    if more_results is not None:
        for key, value in more_results.items():
            all_more_results[key].extend(value)
    nonlocal results
    results = self._format_results(all_candidate_params, n_splits, all_out, all_more_results)
    return results