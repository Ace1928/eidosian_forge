import logging
import os
from packaging import version
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional
from ray._private.dict import flatten_dict
def log_params(self, params_to_log: Dict, run_id: Optional[str]=None):
    """Logs the provided parameters to the run specified by run_id.

        If no ``run_id`` is passed in, then logs to the current active run.
        If there is not active run, then creates a new run and sets it as
        the active run.

        Args:
            params_to_log: Dictionary of parameters to log.
            run_id (Optional[str]): The ID of the run to log to.
        """
    params_to_log = flatten_dict(params_to_log)
    if run_id and self._run_exists(run_id):
        client = self._get_client()
        for key, value in params_to_log.items():
            client.log_param(run_id=run_id, key=key, value=value)
    else:
        for key, value in params_to_log.items():
            self._mlflow.log_param(key=key, value=value)