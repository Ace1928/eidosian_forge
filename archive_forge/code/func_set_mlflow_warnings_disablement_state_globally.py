import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
def set_mlflow_warnings_disablement_state_globally(self, disabled=True):
    """Disables (or re-enables) MLflow warnings globally across all threads.

        Args:
            disabled: If `True`, disables MLflow warnings globally across all threads.
                If `False`, enables MLflow warnings globally across all threads.

        """
    with self._state_lock:
        self._mlflow_warnings_disabled_globally = disabled
        self._modify_patch_state_if_necessary()