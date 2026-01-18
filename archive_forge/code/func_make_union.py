from collections import defaultdict
from itertools import islice
import numpy as np
from scipy import sparse
from .base import TransformerMixin, _fit_context, clone
from .exceptions import NotFittedError
from .preprocessing import FunctionTransformer
from .utils import Bunch, _print_elapsed_time
from .utils._estimator_html_repr import _VisualBlock
from .utils._metadata_requests import METHODS
from .utils._param_validation import HasMethods, Hidden
from .utils._set_output import (
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _BaseComposition, available_if
from .utils.parallel import Parallel, delayed
from .utils.validation import check_is_fitted, check_memory
def make_union(*transformers, n_jobs=None, verbose=False):
    """Construct a :class:`FeatureUnion` from the given transformers.

    This is a shorthand for the :class:`FeatureUnion` constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting.

    Parameters
    ----------
    *transformers : list of estimators
        One or more estimators.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    f : FeatureUnion
        A :class:`FeatureUnion` object for concatenating the results of multiple
        transformer objects.

    See Also
    --------
    FeatureUnion : Class for concatenating the results of multiple transformer
        objects.

    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.pipeline import make_union
    >>> make_union(PCA(), TruncatedSVD())
     FeatureUnion(transformer_list=[('pca', PCA()),
                                   ('truncatedsvd', TruncatedSVD())])
    """
    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)