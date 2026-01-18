import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def set_network(self, machines: Union[List[str], Set[str], str], local_listen_port: int=12400, listen_time_out: int=120, num_machines: int=1) -> 'Booster':
    """Set the network configuration.

        Parameters
        ----------
        machines : list, set or str
            Names of machines.
        local_listen_port : int, optional (default=12400)
            TCP listen port for local machines.
        listen_time_out : int, optional (default=120)
            Socket time-out in minutes.
        num_machines : int, optional (default=1)
            The number of machines for distributed learning application.

        Returns
        -------
        self : Booster
            Booster with set network.
        """
    if isinstance(machines, (list, set)):
        machines = ','.join(machines)
    _safe_call(_LIB.LGBM_NetworkInit(_c_str(machines), ctypes.c_int(local_listen_port), ctypes.c_int(listen_time_out), ctypes.c_int(num_machines)))
    self._network = True
    return self