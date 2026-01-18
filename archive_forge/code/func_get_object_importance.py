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
def get_object_importance(self, pool, train_pool, top_size=-1, type='Average', update_method='SinglePoint', importance_values_sign='All', thread_count=-1, verbose=False, ostr_type=None, log_cout=None, log_cerr=None):
    """
        This is the implementation of the LeafInfluence algorithm from the following paper:
        https://arxiv.org/pdf/1802.06640.pdf

        Parameters
        ----------
        pool : Pool
            The pool for which you want to evaluate the object importances.

        train_pool : Pool
            The pool on which the model has been trained.

        top_size : int (default=-1)
            Method returns the result of the top_size most important train objects.
            If -1, then the top size is not limited.

        type : string, optional (default='Average')
            Possible values:
                - Average (Method returns the mean train objects scores for all input objects)
                - PerObject (Method returns the train objects scores for every input object)

        importance_values_sign : string, optional (default='All')
            Method returns only Positive, Negative or All values.
            Possible values:
                - Positive
                - Negative
                - All

        update_method : string, optional (default='SinglePoint')
            Possible values:
                - SinglePoint
                - TopKLeaves (It is posible to set top size : TopKLeaves:top=2)
                - AllPoints
            Description of the update set methods are given in section 3.1.3 of the paper.

        thread_count : int, optional (default=-1)
            Number of threads.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool or int
            If False, then evaluation is not logged. If True, then each possible iteration is logged.
            If a positive integer, then it stands for the size of batch N. After processing each batch, print progress
            and remaining time.

        ostr_type : string, deprecated, use type instead

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        object_importances : tuple of two arrays (indices and scores) of shape = [top_size]
        """
    if self.is_fitted() and (not self._object._is_oblivious()):
        raise CatBoostError('Object importance is not supported for non symmetric trees')
    if not isinstance(verbose, bool) and (not isinstance(verbose, int)):
        raise CatBoostError('verbose should be bool or int.')
    verbose = int(verbose)
    if verbose < 0:
        raise CatBoostError('verbose should be non-negative.')
    if ostr_type is not None:
        type = ostr_type
        warnings.warn("'ostr_type' parameter will be deprecated soon, use 'type' parameter instead")
    with log_fixup(log_cout, log_cerr):
        result = self._calc_ostr(train_pool, pool, top_size, type, update_method, importance_values_sign, thread_count, verbose)
    return result