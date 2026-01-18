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
def borrow_optimizer(self, shared_module):
    """Borrows optimizer from a shared module. Used in bucketing, where exactly the same
        optimizer (esp. kvstore) is used.

        Parameters
        ----------
        shared_module : Module
        """
    assert shared_module.optimizer_initialized
    self._optimizer = shared_module._optimizer
    self._kvstore = shared_module._kvstore
    self._update_on_kvstore = shared_module._update_on_kvstore
    self._updater = shared_module._updater
    self.optimizer_initialized = True