import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
@validate_params({'estimator': [HasMethods(['fit', 'predict'])], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'groups': ['array-like', None], 'cv': ['cv_object'], 'n_jobs': [Integral, None], 'verbose': ['verbose'], 'fit_params': [dict, None], 'params': [dict, None], 'pre_dispatch': [Integral, str, None], 'method': [StrOptions({'predict', 'predict_proba', 'predict_log_proba', 'decision_function'})]}, prefer_skip_nested_validation=False)
def cross_val_predict(estimator, X, y=None, *, groups=None, cv=None, n_jobs=None, verbose=0, fit_params=None, params=None, pre_dispatch='2*n_jobs', method='predict'):
    """Generate cross-validated estimates for each input data point.

    The data is split according to the cv parameter. Each sample belongs
    to exactly one test set, and its prediction is computed with an
    estimator fitted on the corresponding training set.

    Passing these predictions into an evaluation metric may not be a valid
    way to measure generalization performance. Results can differ from
    :func:`cross_validate` and :func:`cross_val_score` unless all tests sets
    have equal size and the metric decomposes over samples.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator
        The estimator instance to use to fit the data. It must implement a `fit`
        method and the method given by the `method` parameter.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs),             default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

        .. versionchanged:: 1.4
            ``groups`` can only be passed if metadata routing is not enabled
            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing
            is enabled, pass ``groups`` alongside other metadata via the ``params``
            argument instead. E.g.:
            ``cross_val_predict(..., params={'groups': groups})``.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and
        predicting are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

        .. deprecated:: 1.4
            This parameter is deprecated and will be removed in version 1.6. Use
            ``params`` instead.

    params : dict, default=None
        Parameters to pass to the underlying estimator's ``fit`` and the CV
        splitter.

        .. versionadded:: 1.4

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : {'predict', 'predict_proba', 'predict_log_proba',               'decision_function'}, default='predict'
        The method to be invoked by `estimator`.

    Returns
    -------
    predictions : ndarray
        This is the result of calling `method`. Shape:

            - When `method` is 'predict' and in special case where `method` is
              'decision_function' and the target is binary: (n_samples,)
            - When `method` is one of {'predict_proba', 'predict_log_proba',
              'decision_function'} (unless special case above):
              (n_samples, n_classes)
            - If `estimator` is :term:`multioutput`, an extra dimension
              'n_outputs' is added to the end of each shape above.

    See Also
    --------
    cross_val_score : Calculate score for each CV split.
    cross_validate : Calculate one or more scores and timings for each CV
        split.

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    params = _check_params_groups_deprecation(fit_params, params, groups)
    X, y = indexable(X, y)
    if _routing_enabled():
        router = MetadataRouter(owner='cross_validate').add(splitter=cv, method_mapping=MethodMapping().add(caller='fit', callee='split')).add(estimator=estimator, method_mapping=MethodMapping().add(caller='fit', callee='fit'))
        try:
            routed_params = process_routing(router, 'fit', **params)
        except UnsetMetadataPassedError as e:
            unrequested_params = sorted(e.unrequested_params)
            raise UnsetMetadataPassedError(message=f"{unrequested_params} are passed to `cross_val_predict` but are not explicitly set as requested or not requested for cross_validate's estimator: {estimator.__class__.__name__} Call `.set_fit_request({{{{metadata}}}}=True)` on the estimator for each metadata in {unrequested_params} that you want to use and `metadata=False` for not using it. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.", unrequested_params=e.unrequested_params, routed_params=e.routed_params)
    else:
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={'groups': groups})
        routed_params.estimator = Bunch(fit=params)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, **routed_params.splitter.split))
    test_indices = np.concatenate([test for _, test in splits])
    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')
    encode = method in ['decision_function', 'predict_proba', 'predict_log_proba'] and y is not None
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    predictions = parallel((delayed(_fit_and_predict)(clone(estimator), X, y, train, test, routed_params.estimator.fit, method) for train, test in splits))
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)
    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]