import logging
import copy
from ..initializer import Uniform
from .base_module import BaseModule
def get_input_grads(self, merge_multi_context=True):
    """Gets the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArrays or list of list of NDArrays
            If `merge_multi_context` is ``True``, it is like ``[grad1, grad2]``. Otherwise, it
            is like ``[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]``. All the output
            elements are `NDArray`.
        """
    assert self.binded and self.params_initialized and self.inputs_need_grad
    return self._modules[0].get_input_grads(merge_multi_context=merge_multi_context)