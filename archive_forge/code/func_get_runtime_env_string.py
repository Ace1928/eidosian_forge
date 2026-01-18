import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_runtime_env_string(self):
    """Get the runtime env string used for the current driver or worker.

        Returns:
            The runtime env string currently using by this worker.
        """
    return self.worker.runtime_env