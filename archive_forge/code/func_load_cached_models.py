import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
def load_cached_models(eight_schools_data, draws, chains, libs=None):
    """Load pystan, emcee, and pyro models from pickle."""
    here = os.path.dirname(os.path.abspath(__file__))
    supported = (('pystan', pystan_noncentered_schools), ('emcee', emcee_schools_model), ('pyro', pyro_noncentered_schools), ('numpyro', numpyro_schools_model))
    data_directory = os.path.join(here, 'saved_models')
    models = {}
    if isinstance(libs, str):
        libs = [libs]
    for library_name, func in supported:
        if libs is not None and library_name not in libs:
            continue
        library = library_handle(library_name)
        if library.__name__ == 'stan':
            _log.info('Generating and loading stan model')
            models['pystan'] = func(eight_schools_data, draws, chains)
            continue
        py_version = sys.version_info
        fname = '{0.major}.{0.minor}_{1.__name__}_{1.__version__}_{2}_{3}_{4}.pkl.gzip'.format(py_version, library, sys.platform, draws, chains)
        path = os.path.join(data_directory, fname)
        if not os.path.exists(path):
            with gzip.open(path, 'wb') as buff:
                try:
                    _log.info('Generating and caching %s', fname)
                    cloudpickle.dump(func(eight_schools_data, draws, chains), buff)
                except AttributeError as err:
                    raise AttributeError(f'Failed caching {library_name}') from err
        with gzip.open(path, 'rb') as buff:
            _log.info('Loading %s from cache', fname)
            models[library.__name__] = cloudpickle.load(buff)
    return models