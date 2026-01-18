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
def get_feature_importance(self, data=None, type=EFstrType.FeatureImportance, prettified=False, thread_count=-1, verbose=False, fstr_type=None, shap_mode='Auto', model_output='Raw', interaction_indices=None, shap_calc_type='Regular', reference_data=None, sage_n_samples=128, sage_batch_size=512, sage_detect_convergence=True, log_cout=None, log_cerr=None):
    """
        Parameters
        ----------
        data :
            Data to get feature importance.
            If type in ('LossFunctionChange', 'ShapValues', 'ShapInteractionValues') data must of Pool type.
                For every object in this dataset feature importances will be calculated.
            if type == 'SageValues' data must of Pool type.
                For every feature in this dataset importance will be calculated.
            If type == 'PredictionValuesChange', data is None or a dataset of Pool type
                Dataset specification is needed only in case if the model does not contain leaf weight information (trained with CatBoost v < 0.9).
            If type == 'PredictionDiff' data must contain a matrix of feature values of shape (2, n_features).
                Possible types are catboost.Pool or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData or pandas.SparseDataFrame or scipy.sparse.spmatrix
            If type == 'FeatureImportance'
                See 'PredictionValuesChange' for non-ranking metrics and 'LossFunctionChange' for ranking metrics.
            If type == 'Interaction'
                This parameter is not used.

        type : EFstrType or string (converted to EFstrType), optional
                    (default=EFstrType.FeatureImportance)
            Possible values:
                - PredictionValuesChange
                    Calculate score for every feature.
                - LossFunctionChange
                    Calculate score for every feature by loss.
                - FeatureImportance
                    PredictionValuesChange for non-ranking metrics and LossFunctionChange for ranking metrics
                - ShapValues
                    Calculate SHAP Values for every object.
                - ShapInteractionValues
                    Calculate SHAP Interaction Values between each pair of features for every object
                - Interaction
                    Calculate pairwise score between every feature.
                - PredictionDiff
                    Calculate most important features explaining difference in predictions for a pair of documents.
                - SageValues
                    Calculate SAGE value for every feature

        prettified : bool, optional (default=False)
            change returned data format to the list of (feature_id, importance) pairs sorted by importance

        thread_count : int, optional (default=-1)
            Number of threads.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool or int
            If False, then evaluation is not logged. If True, then each possible iteration is logged.
            If a positive integer, then it stands for the size of batch N. After processing each batch, print progress
            and remaining time.

        fstr_type : string, deprecated, use type instead

        shap_mode : string, optional (default="Auto")
            used only for ShapValues type
            Possible values:
                - "Auto"
                    Use direct SHAP Values calculation only if data size is smaller than average leaves number
                    (the best of two strategies below is chosen).
                - "UsePreCalc"
                    Calculate SHAP Values for every leaf in preprocessing. Final complexity is
                    O(NT(D+F))+O(TL^2 D^2) where N is the number of documents(objects), T - number of trees,
                    D - average tree depth, F - average number of features in tree, L - average number of leaves in tree
                    This is much faster (because of a smaller constant) than direct calculation when N >> L
                - "NoPreCalc"
                    Use direct SHAP Values calculation calculation with complexity O(NTLD^2). Direct algorithm
                    is faster when N < L (algorithm from https://arxiv.org/abs/1802.03888)

        shap_calc_type : EShapCalcType or string, optional (default="Regular")
            used only for ShapValues type
            Possible values:
                - "Regular"
                    Calculate regular SHAP values
                - "Approximate"
                    Calculate approximate SHAP values
                - "Exact"
                    Calculate exact SHAP values

        interaction_indices : list of int or string (feature_idx_1, feature_idx_2), optional (default=None)
            used only for ShapInteractionValues type
            Calculate SHAP Interaction Values between pair of features feature_idx_1 and feature_idx_2 for every object

        reference_data: catboost.Pool or None
            Reference data for Independent Tree SHAP values from https://arxiv.org/abs/1905.04610v1
            if type == 'ShapValues' and reference_data is not None, then Independent Tree SHAP values are calculated

        sage_n_samples: int, optional (default=32)
            Number of outer samples used in SAGE values approximation algorithm
        sage_batch_size: int, optional (default=min(512, number of samples in dataset))
            Number of samples used on each step of SAGE values approximation algorithm
        sage_detect_convergence: bool, optional (default=False)
            If set True, sage values calculation will be stopped either when sage values converge
            or when sage_n_samples iterations of algorithm pass

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        depends on type:
            - FeatureImportance
                See PredictionValuesChange for non-ranking metrics and LossFunctionChange for ranking metrics.
            - PredictionValuesChange, LossFunctionChange, PredictionDiff, SageValues with prettified=False (default)
                list of length [n_features] with feature_importance values (float) for feature
            - PredictionValuesChange, LossFunctionChange, PredictionDiff, SageValues with prettified=True
                list of length [n_features] with (feature_id (string), feature_importance (float)) pairs, sorted by feature_importance in descending order
            - ShapValues
                np.ndarray of shape (n_objects, n_features + 1) with Shap values (float) for (object, feature).
                In case of multiclass the returned value is np.ndarray of shape
                (n_objects, classes_count, n_features + 1). For each object it contains Shap values (float).
                Values are calculated for RawFormulaVal predictions.
            - ShapInteractionValues
                np.ndarray of shape (n_objects, n_features + 1, n_features + 1) with Shap interaction values (float) for (object, feature(i), feature(j)).
                In case of multiclass the returned value is np.ndarray of shape
                (n_objects, classes_count, n_features + 1, n_features + 1). For each object it contains Shap interaction values (float).
                Values are calculated for RawFormulaVal predictions.
            - Interaction
                list of length [n_features] of 3-element lists of (first_feature_index, second_feature_index, interaction_score (float))
        """
    with log_fixup(log_cout, log_cerr):
        if not isinstance(verbose, bool) and (not isinstance(verbose, int)):
            raise CatBoostError('verbose should be bool or int.')
        verbose = int(verbose)
        if verbose < 0:
            raise CatBoostError('verbose should be non-negative.')
        if fstr_type is not None:
            type = fstr_type
        type = enum_from_enum_or_str(EFstrType, type)
        if type == EFstrType.FeatureImportance:
            loss = self._object._get_loss_function_name()
            if loss and is_groupwise_metric(loss):
                type = EFstrType.LossFunctionChange
            else:
                type = EFstrType.PredictionValuesChange
        if type == EFstrType.PredictionDiff:
            data, _ = self._process_predict_input_data(data, 'get_feature_importance', thread_count)
            if data.num_row() != 2:
                raise CatBoostError('{} requires a pair of documents, found {}'.format(type, data.num_row()))
        elif data is not None and (not isinstance(data, Pool)):
            raise CatBoostError('Invalid data type={}, must be catboost.Pool.'.format(_typeof(data)))
        need_meta_info = type == EFstrType.PredictionValuesChange
        empty_data_is_ok = need_meta_info and self._object._has_leaf_weights_in_model() or type == EFstrType.Interaction
        if not empty_data_is_ok:
            if data is None:
                if need_meta_info:
                    raise CatBoostError('Model has no meta information needed to calculate feature importances.                             Pass training dataset to this function.')
                else:
                    raise CatBoostError('Feature importance type {} requires training dataset                             to be passed to this function.'.format(type))
            if data.is_empty_:
                raise CatBoostError('data is empty.')
        shap_calc_type = enum_from_enum_or_str(EShapCalcType, shap_calc_type).value
        fstr, feature_names = self._calc_fstr(type, data, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type, sage_n_samples, sage_batch_size, sage_detect_convergence)
        if type in (EFstrType.PredictionValuesChange, EFstrType.LossFunctionChange, EFstrType.PredictionDiff, EFstrType.SageValues):
            feature_importances = [value[0] for value in fstr]
            attribute_name = None
            if type == EFstrType.PredictionValuesChange:
                attribute_name = '_prediction_values_change'
            if type == EFstrType.LossFunctionChange:
                attribute_name = '_loss_value_change'
            if attribute_name:
                setattr(self, attribute_name, feature_importances)
            if prettified:
                feature_importances = sorted(zip(feature_names, feature_importances), key=itemgetter(1), reverse=True)
                columns = ['Feature Id', 'Importances']
                return DataFrame(feature_importances, columns=columns)
            else:
                return np.array(feature_importances)
        elif type == EFstrType.ShapValues:
            if isinstance(fstr[0][0], ARRAY_TYPES):
                return np.array([np.array([np.array([value for value in dimension]) for dimension in doc]) for doc in fstr])
            else:
                result = [[value for value in doc] for doc in fstr]
                if prettified:
                    return DataFrame(result)
                else:
                    return np.array(result)
        elif type == EFstrType.ShapInteractionValues:
            if isinstance(fstr[0][0], ARRAY_TYPES):
                return np.array([np.array([np.array([feature2 for feature2 in feature1]) for feature1 in doc]) for doc in fstr])
            else:
                return np.array([np.array([np.array([np.array([feature2 for feature2 in feature1]) for feature1 in dimension]) for dimension in doc]) for doc in fstr])
        elif type == EFstrType.Interaction:
            result = [[int(row[0]), int(row[1]), row[2]] for row in fstr]
            if prettified:
                columns = ['First Feature Index', 'Second Feature Index', 'Interaction']
                return DataFrame(result, columns=columns)
            else:
                return np.array(result)