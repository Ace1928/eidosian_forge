import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
def set_non_mlflow_warnings_rerouting_state_for_current_thread(self, rerouted=True):
    """Enables (or disables) rerouting of non-MLflow warnings to an MLflow event logger with
        level WARNING (e.g. `logger.warning()`) for the current thread.

        Args:
            rerouted: If `True`, enables non-MLflow warning rerouting for the current thread.
                If `False`, disables non-MLflow warning rerouting for the current thread.
                non-MLflow warning behavior in other threads is unaffected.

        """
    with self._state_lock:
        if rerouted:
            self._rerouted_threads.add(get_current_thread_id())
        else:
            self._rerouted_threads.discard(get_current_thread_id())
        self._modify_patch_state_if_necessary()