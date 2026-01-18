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
@validate_params({'estimator': [HasMethods('fit')], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'groups': ['array-like', None], 'scoring': [StrOptions(set(get_scorer_names())), callable, list, tuple, dict, None], 'cv': ['cv_object'], 'n_jobs': [Integral, None], 'verbose': ['verbose'], 'fit_params': [dict, None], 'params': [dict, None], 'pre_dispatch': [Integral, str], 'return_train_score': ['boolean'], 'return_estimator': ['boolean'], 'return_indices': ['boolean'], 'error_score': [StrOptions({'raise'}), Real]}, prefer_skip_nested_validation=False)
def cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, params=None, pre_dispatch='2*n_jobs', return_train_score=False, return_estimator=False, return_indices=False, error_score=np.nan):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
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
            ``cross_validate(..., params={'groups': groups})``.

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
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
        Parameters to pass to the underlying estimator's ``fit``, the scorer,
        and the CV splitter.

        .. versionadded:: 1.4

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

        .. versionadded:: 0.20

    return_indices : bool, default=False
        Whether to return the train-test indices selected for each split.

        .. versionadded:: 1.3

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.
            ``indices``
                The train/test positional indices for each cv split. A dictionary
                is returned where the keys are either `"train"` or `"test"`
                and the associated values are a list of integer-dtyped NumPy
                arrays with the indices. Available only if `return_indices=True`.

    See Also
    --------
    cross_val_score : Run cross-validation for single metric evaluation.

    cross_val_predict : Get predictions from each split of cross-validation for
        diagnostic purposes.

    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()

    Single metric evaluation using ``cross_validate``

    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.3315057 , 0.08022103, 0.03531816])

    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)

    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    >>> print(scores['train_r2'])
    [0.28009951 0.3908844  0.22784907]
    """
    params = _check_params_groups_deprecation(fit_params, params, groups)
    X, y = indexable(X, y)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)
    if _routing_enabled():
        if isinstance(scorers, dict):
            _scorer = _MultimetricScorer(scorers=scorers, raise_exc=error_score == 'raise')
        else:
            _scorer = scorers
        router = MetadataRouter(owner='cross_validate').add(splitter=cv, method_mapping=MethodMapping().add(caller='fit', callee='split')).add(estimator=estimator, method_mapping=MethodMapping().add(caller='fit', callee='fit')).add(scorer=_scorer, method_mapping=MethodMapping().add(caller='fit', callee='score'))
        try:
            routed_params = process_routing(router, 'fit', **params)
        except UnsetMetadataPassedError as e:
            unrequested_params = sorted(e.unrequested_params)
            raise UnsetMetadataPassedError(message=f"{unrequested_params} are passed to cross validation but are not explicitly set as requested or not requested for cross_validate's estimator: {estimator.__class__.__name__}. Call `.set_fit_request({{{{metadata}}}}=True)` on the estimator for each metadata in {unrequested_params} that you want to use and `metadata=False` for not using it. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.", unrequested_params=e.unrequested_params, routed_params=e.routed_params)
    else:
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={'groups': groups})
        routed_params.estimator = Bunch(fit=params)
        routed_params.scorer = Bunch(score={})
    indices = cv.split(X, y, **routed_params.splitter.split)
    if return_indices:
        indices = list(indices)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel((delayed(_fit_and_score)(clone(estimator), X, y, scorer=scorers, train=train, test=test, verbose=verbose, parameters=None, fit_params=routed_params.estimator.fit, score_params=routed_params.scorer.score, return_train_score=return_train_score, return_times=True, return_estimator=return_estimator, error_score=error_score) for train, test in indices))
    _warn_or_raise_about_fit_failures(results, error_score)
    if callable(scoring):
        _insert_error_scores(results, error_score)
    results = _aggregate_score_dicts(results)
    ret = {}
    ret['fit_time'] = results['fit_time']
    ret['score_time'] = results['score_time']
    if return_estimator:
        ret['estimator'] = results['estimator']
    if return_indices:
        ret['indices'] = {}
        ret['indices']['train'], ret['indices']['test'] = zip(*indices)
    test_scores_dict = _normalize_score_results(results['test_scores'])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results['train_scores'])
    for name in test_scores_dict:
        ret['test_%s' % name] = test_scores_dict[name]
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = train_scores_dict[name]
    return ret