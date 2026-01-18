import logging
import warnings
from .. import context as ctx
from .. import optimizer as opt
from .. import ndarray as nd
from .executor_group import DataParallelExecutorGroup
from ..model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from ..model import load_checkpoint
from ..initializer import Uniform, InitDesc
from ..io import DataDesc
from ..ndarray import zeros
from .base_module import BaseModule, _check_input_names, _parse_data_desc
@property
def label_names(self):
    """A list of names for labels required by this module."""
    return self._label_names