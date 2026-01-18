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
def save_optimizer_states(self, fname):
    """Saves optimizer (updater) state to a file.

        Parameters
        ----------
        fname : str
            Path to output states file.
        """
    assert self.optimizer_initialized
    if self._update_on_kvstore:
        self._kvstore.save_optimizer_states(fname)
    else:
        with open(fname, 'wb') as fout:
            fout.write(self._updater.get_states())