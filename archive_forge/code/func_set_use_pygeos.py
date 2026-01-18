import contextlib
from packaging.version import Version
import importlib
import os
import warnings
import numpy as np
import pandas as pd
import shapely
import shapely.geos
def set_use_pygeos(val=None):
    """
    Set the global configuration on whether to use PyGEOS or not.

    The default is use PyGEOS if it is installed. This can be overridden
    with an environment variable USE_PYGEOS (this is only checked at
    first import, cannot be changed during interactive session).

    Alternatively, pass a value here to force a True/False value.
    """
    global USE_PYGEOS
    global USE_SHAPELY_20
    global PYGEOS_SHAPELY_COMPAT
    env_use_pygeos = os.getenv('USE_PYGEOS', None)
    if val is not None:
        USE_PYGEOS = bool(val)
    elif USE_PYGEOS is None:
        if SHAPELY_GE_20:
            USE_PYGEOS = False
        else:
            USE_PYGEOS = HAS_PYGEOS
        if env_use_pygeos is not None:
            USE_PYGEOS = bool(int(env_use_pygeos))
    if USE_PYGEOS:
        try:
            import pygeos
            if not Version(pygeos.__version__) >= Version('0.8'):
                if SHAPELY_GE_20:
                    USE_PYGEOS = False
                    warnings.warn('The PyGEOS version is too old, and Shapely >= 2 is installed, thus using Shapely by default and not PyGEOS.', stacklevel=2)
                else:
                    raise ImportError('PyGEOS >= 0.8 is required, version {0} is installed'.format(pygeos.__version__))
            from shapely.geos import geos_version_string as shapely_geos_version
            from pygeos import geos_capi_version_string
            if not shapely_geos_version.startswith(geos_capi_version_string):
                warnings.warn('The Shapely GEOS version ({}) is incompatible with the GEOS version PyGEOS was compiled with ({}). Conversions between both will be slow.'.format(shapely_geos_version, geos_capi_version_string), stacklevel=2)
                PYGEOS_SHAPELY_COMPAT = False
            else:
                PYGEOS_SHAPELY_COMPAT = True
        except ImportError:
            raise ImportError(INSTALL_PYGEOS_ERROR)
    if USE_PYGEOS:
        warnings.warn('GeoPandas is set to use PyGEOS over Shapely. PyGEOS support is deprecatedand will be removed in GeoPandas 1.0, released in the Q1 of 2024. Please migrate to Shapely 2.0 (https://geopandas.org/en/stable/docs/user_guide/pygeos_to_shapely.html).', DeprecationWarning, stacklevel=6)
    USE_SHAPELY_20 = not USE_PYGEOS and SHAPELY_GE_20