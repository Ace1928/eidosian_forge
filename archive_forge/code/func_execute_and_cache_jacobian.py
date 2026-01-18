import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
def execute_and_cache_jacobian(self, tapes: Batch):
    """Forward pass used to cache the results and jacobians.

        Args:
            tapes (tuple[`~.QuantumScript`]): the batch of tapes to execute and take derivatives of

        Returns:
            ResultBatch: the results of the execution.

        Side Effects:
            Caches both the results and jacobian into ``_results_cache`` and ``_jacs_cache``.

        """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Forward pass called with %s', tapes)
    results, jac = self._dev_execute_and_compute_derivatives(tapes)
    self._results_cache[tapes] = results
    self._jacs_cache[tapes] = jac
    return results