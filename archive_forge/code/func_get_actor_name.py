import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_actor_name(self) -> Optional[str]:
    """Get the current actor name of this worker.

        This shouldn't be used in a driver process.
        The name is in string format.

        Returns:
            The current actor name of this worker.
            If a current worker is an actor, and
            if actor name doesn't exist, it returns an empty string.
            If a current worker is not an actor, it returns None.
        """
    if self.worker.mode != ray._private.worker.WORKER_MODE:
        logger.warning(f'This method is only available when the process is a worker. Current mode: {self.worker.mode}')
    actor_id = self.worker.actor_id
    return self.worker.actor_name if not actor_id.is_nil() else None