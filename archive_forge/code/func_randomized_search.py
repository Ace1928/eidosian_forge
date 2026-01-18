from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def randomized_search(self, param_distributions, X, y=None, cv=3, n_iter=10, partition_random_seed=0, calc_cv_statistics=True, search_by_train_test_split=True, refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=True, plot=False, plot_file=None, log_cout=None, log_cerr=None):
    """
        Randomized search on hyper parameters.
        After calling this method model is fitted and can be used, if not specified otherwise (refit=False).

        In contrast to grid_search, not all parameter values are tried out,
        but rather a fixed number of parameter settings is sampled from the specified distributions.
        The number of parameter settings that are tried is given by n_iter.

        Parameters
        ----------
        param_distributions: dict
            Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
            Distributions must provide a rvs method for sampling (such as those from scipy.stats.distributions).
            If a list is given, it is sampled uniformly.

        X: numpy.ndarray or pandas.DataFrame or catboost.Pool
            Data to compute statistics on

        y: list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with either:
              - numerical values - for regression (including multiregression), ranking and binary classification problems
              - class labels (boolean, integer or string) - for classification (including multiclassification) problems
            Use only if X is not catboost.Pool and does not point to a file.

        cv: int, cross-validation generator or an iterable, optional (default=None)
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a (Stratified)KFold
            - one of the scikit-learn splitter classes
                (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            - An iterable yielding (train, test) splits as arrays of indices.

        n_iter: int
            Number of parameter settings that are sampled.
            n_iter trades off runtime vs quality of the solution.

        partition_random_seed: int, optional (default=0)
            Use this as the seed value for random permutation of the data.
            Permutation is performed before splitting the data for cross validation.
            Each seed generates unique data splits.
            Used only when cv is None or int.

        search_by_train_test_split: bool, optional (default=True)
            If True, source dataset is splitted into train and test parts, models are trained
            on the train part and parameters are compared by loss function score on the test part.
            After that, if calc_cv_statistics=true, statistics on metrics are calculated
            using cross-validation using best parameters and the model is fitted with these parameters.

            If False, every iteration of grid search evaluates results on cross-validation.
            It is recommended to set parameter to True for large datasets, and to False for small datasets.

        calc_cv_statistics: bool, optional (default=True)
            The parameter determines whether quality should be estimated.
            using cross-validation with the found best parameters. Used only when search_by_train_test_split=True.

        refit: bool (default=True)
            Refit an estimator using the best found parameters on the whole dataset.

        shuffle: bool, optional (default=True)
            Shuffle the dataset objects before parameters searching.

        stratified: bool, optional (default=None)
            Perform stratified sampling. True for classification and False otherwise.
            Currently supported only for cross-validation.

        train_size: float, optional (default=0.8)
            Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.

        verbose: bool or int, optional (default=True)
            If verbose is int, it determines the frequency of writing metrics to output
            verbose==True is equal to verbose==1
            When verbose==False, there is no messages

        plot : bool, optional (default=False)
            If True, draw train and eval error for every set of parameters in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error for every set of parameters to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        dict with two fields:
            'params': dict of best found parameters
            'cv_results': dict or pandas.core.frame.DataFrame with cross-validation results
                columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
        """
    if n_iter <= 0:
        assert CatBoostError('n_iter should be a positive number')
    if not isinstance(param_distributions, Mapping):
        assert CatBoostError('param_distributions should be a dictionary')
    for key in param_distributions:
        if not isinstance(param_distributions[key], Iterable) and (not hasattr(param_distributions[key], 'rvs')):
            raise TypeError("Parameter grid value is not iterable and do not have 'rvs' method (key={!r}, value={!r})".format(key, param_distributions[key]))
    return self._tune_hyperparams(param_grid=param_distributions, X=X, y=y, cv=cv, n_iter=n_iter, partition_random_seed=partition_random_seed, calc_cv_statistics=calc_cv_statistics, search_by_train_test_split=search_by_train_test_split, refit=refit, shuffle=shuffle, stratified=stratified, train_size=train_size, verbose=verbose, plot=plot, plot_file=plot_file, log_cout=log_cout, log_cerr=log_cerr)