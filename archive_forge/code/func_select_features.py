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
def select_features(self, X, y=None, eval_set=None, features_for_select=None, num_features_to_select=None, algorithm=None, steps=None, shap_calc_type=None, train_final_model=True, verbose=None, logging_level=None, plot=False, plot_file=None, log_cout=None, log_cerr=None, grouping=None, features_tags_for_select=None, num_features_tags_to_select=None):
    """
        Select best features from pool according to loss value.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with either:
              - numerical values - for regression (including multiregression), ranking and binary classification problems
              - class labels (boolean, integer or string) - for classification (including multiclassification) problems
            Use only if X is not catboost.Pool and does not point to a file.

        eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
            Validation dataset or datasets for metrics calculation and possibly early stopping.

        features_for_select : str or list of feature indices, names or ranges
            (for grouping = Individual)
            Which features should participate in the selection.
            Format examples:
                - [0, 2, 3, 4, 17]
                - [0, "2-4", 17] (both ends in ranges are inclusive)
                - "0,2-4,20"
                - ["Name0", "Name2", "Name3", "Name4", "Name20"]

        num_features_to_select : positive int
            (for grouping = Individual)
            How many features to select from features_for_select.

        algorithm : EFeaturesSelectionAlgorithm or string, optional (default=RecursiveByShapValues)
            Which algorithm to use for features selection.
            Possible values:
                - RecursiveByPredictionValuesChange
                    Use prediction values change as feature strength, eliminate batch of features at once.
                - RecursiveByLossFunctionChange
                    Use loss function change as feature strength, eliminate batch of features at each step.
                - RecursiveByShapValues
                    Use shap values to estimate loss function change, eliminate features one by one.

        steps : positive int, optional (default=1)
            How many steps should be performed. In other words, how many times a full model will be trained.
            More steps give more accurate results.

        shap_calc_type : EShapCalcType or string, optional (default=Regular)
            Which method to use for calculation of shap values.
            Possible values:
                - Regular
                    Calculate regular SHAP values
                - Approximate
                    Calculate approximate SHAP values
                - Exact
                    Calculate exact SHAP values

        train_final_model : bool, optional (default=True)
            Need to fit model with selected features.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook.

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        grouping : EFeaturesSelectionGrouping or string, optional (default=Individual)
            Which grouping to use for features selection.
            Possible values:
                - Individual
                    Select individual features
                - ByTags
                    Select feature groups (marked by tags)

        features_tags_for_select : list of strings
            (for grouping = ByTags)
            Which features tags should participate in the selection.

        num_features_tags_to_select : positive int
            (for grouping = ByTags)
            How many features tags to select from features_tags_for_select.

        Returns
        -------
        dict with fields:
            'selected_features': list of selected features indices
            'eliminated_features': list of eliminated features indices
            'selected_features_tags': list of selected features tags (optional, present if grouping == ByTags)
            'eliminated_features_tags': list of selected features tags (optional, present if grouping == ByTags)
        """
    if train_final_model and self.is_fitted():
        raise CatBoostError('Model was already fitted. Set train_final_model to False or use not fitted model.')
    if X is None:
        raise CatBoostError('X must not be None')
    if y is None and (not isinstance(X, PATH_TYPES + (Pool,))):
        raise CatBoostError('y may be None only when X is an instance of catboost.Pool, str or pathlib.Path.')
    with log_fixup(log_cout, log_cerr):
        train_params = self._prepare_train_params(X=X, y=y, eval_set=eval_set, verbose=verbose, logging_level=logging_level)
        params = train_params['params']
        if grouping is None:
            grouping = EFeaturesSelectionGrouping.Individual
        else:
            grouping = enum_from_enum_or_str(EFeaturesSelectionGrouping, grouping).value
            params['features_selection_grouping'] = grouping
        if grouping == EFeaturesSelectionGrouping.Individual:
            if isinstance(features_for_select, Iterable) and (not isinstance(features_for_select, STRING_TYPES)):
                features_for_select = ','.join(map(str, features_for_select))
            if features_for_select is None:
                raise CatBoostError('You should specify features_for_select')
            if features_tags_for_select is not None:
                raise CatBoostError('You should not specify features_tags_for_select when grouping is Individual')
            if num_features_to_select is None:
                raise CatBoostError('You should specify num_features_to_select')
            if num_features_tags_to_select is not None:
                raise CatBoostError('You should not specify num_features_tags_to_select when grouping is Individual')
            params['features_for_select'] = features_for_select
            params['num_features_to_select'] = num_features_to_select
        else:
            if features_tags_for_select is None:
                raise CatBoostError('You should specify features_tags_for_select')
            if not isinstance(features_tags_for_select, Sequence):
                raise CatBoostError('features_tags_for_select must be a list of strings')
            if features_for_select is not None:
                raise CatBoostError('You should not specify features_for_select when grouping is ByTags')
            if num_features_tags_to_select is None:
                raise CatBoostError('You should specify num_features_tags_to_select')
            if num_features_to_select is not None:
                raise CatBoostError('You should not specify num_features_to_select when grouping is ByTags')
            params['features_tags_for_select'] = features_tags_for_select
            params['num_features_tags_to_select'] = num_features_tags_to_select
        objective = params.get('loss_function')
        is_custom_objective = objective is not None and (not isinstance(objective, string_types))
        if is_custom_objective:
            raise CatBoostError('Custom objective is not supported for features selection')
        if algorithm is not None:
            params['features_selection_algorithm'] = enum_from_enum_or_str(EFeaturesSelectionAlgorithm, algorithm).value
        if steps is not None:
            params['features_selection_steps'] = steps
        if shap_calc_type is not None:
            params['shap_calc_type'] = enum_from_enum_or_str(EShapCalcType, shap_calc_type).value
        if train_final_model:
            params['train_final_model'] = True
        train_pool = train_params['train_pool']
        test_pool = None
        if len(train_params['eval_sets']) > 1:
            raise CatBoostError('Multiple eval sets are not supported for features selection')
        elif len(train_params['eval_sets']) == 1:
            test_pool = train_params['eval_sets'][0]
        train_dir = _get_train_dir(self.get_params())
        create_dir_if_not_exist(train_dir)
        plot_dirs = []
        for step in range(steps or 1):
            plot_dirs.append(os.path.join(train_dir, 'model-{}'.format(step)))
        if train_final_model:
            plot_dirs.append(os.path.join(train_dir, 'model-final'))
        for plot_dir in plot_dirs:
            create_dir_if_not_exist(plot_dir)
        with plot_wrapper(plot, plot_file=plot_file, plot_title='Select features plot', train_dirs=plot_dirs):
            summary = self._object._select_features(train_pool, test_pool, params)
        if train_final_model:
            self._set_trained_model_attributes()
        if plot:
            figures = plot_features_selection_loss_graphs(summary)
            figures['features'].show()
            if 'features_tags' in figures:
                figures['features_tags'].show()
    return summary