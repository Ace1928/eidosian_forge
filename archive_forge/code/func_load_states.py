from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def load_states(self, fname):
    """Loads trainer states (e.g. optimizer, momentum) from a file.

        Parameters
        ----------
        fname : str
            Path to input states file.

        Note
        ----
        `optimizer.param_dict`, which contains Parameter information (such as
        `lr_mult` and `wd_mult`) will not be loaded from the file, but rather set
        based on current Trainer's parameters.
        """
    if not self._kv_initialized:
        self._init_kvstore()
    if self._params_to_init:
        self._init_params()
    if self._update_on_kvstore:
        self._kvstore.load_optimizer_states(fname)
        self._optimizer = self._kvstore._updater.optimizer
    else:
        with open(fname, 'rb') as f:
            states = f.read()
        for updater in self._updaters:
            updater.set_states(states)
            updater.optimizer = self._updaters[0].optimizer
        self._optimizer = self._updaters[0].optimizer
    param_dict = {i: param for i, param in enumerate(self._params)}
    self._optimizer.param_dict = param_dict