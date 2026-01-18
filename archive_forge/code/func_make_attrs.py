import datetime
import functools
import importlib
import re
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np
import tree
import xarray as xr
from .. import __version__, utils
from ..rcparams import rcParams
def make_attrs(attrs=None, library=None):
    """Make standard attributes to attach to xarray datasets.

    Parameters
    ----------
    attrs : dict (optional)
        Additional attributes to add or overwrite

    Returns
    -------
    dict
        attrs
    """
    default_attrs = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'arviz_version': __version__}
    if library is not None:
        library_name = library.__name__
        default_attrs['inference_library'] = library_name
        try:
            version = importlib.metadata.version(library_name)
            default_attrs['inference_library_version'] = version
        except importlib.metadata.PackageNotFoundError:
            if hasattr(library, '__version__'):
                version = library.__version__
                default_attrs['inference_library_version'] = version
    if attrs is not None:
        default_attrs.update(attrs)
    return default_attrs