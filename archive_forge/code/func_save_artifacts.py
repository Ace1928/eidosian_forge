import logging
import os
from packaging import version
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional
from ray._private.dict import flatten_dict
def save_artifacts(self, dir: str, run_id: Optional[str]=None):
    """Saves directory as artifact to the run specified by run_id.

        If no ``run_id`` is passed in, then saves to the current active run.
        If there is not active run, then creates a new run and sets it as
        the active run.

        Args:
            dir: Path to directory containing the files to save.
            run_id (Optional[str]): The ID of the run to log to.
        """
    if run_id and self._run_exists(run_id):
        client = self._get_client()
        client.log_artifacts(run_id=run_id, local_dir=dir)
    else:
        self._mlflow.log_artifacts(local_dir=dir)