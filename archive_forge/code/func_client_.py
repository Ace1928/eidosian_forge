import operator
import socket
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse
import numpy as np
import scipy.sparse as ss
from .basic import LightGBMError, _choose_param_value, _ConfigAliases, _log_info, _log_warning
from .compat import (DASK_INSTALLED, PANDAS_INSTALLED, SKLEARN_INSTALLED, Client, Future, LGBMNotFittedError, concat,
from .sklearn import (LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor, _LGBM_ScikitCustomObjectiveFunction,
@property
def client_(self) -> Client:
    """:obj:`dask.distributed.Client`: Dask client.

        This property can be passed in the constructor or updated
        with ``model.set_params(client=client)``.
        """
    if not getattr(self, 'fitted_', False):
        raise LGBMNotFittedError('Cannot access property client_ before calling fit().')
    return _get_dask_client(client=self.client)