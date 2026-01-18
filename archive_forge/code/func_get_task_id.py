import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_task_id(self) -> Optional[str]:
    """Get current task ID for this worker or driver.

        Task ID is the id of a Ray task. The ID will be in hex format.
        This shouldn't be used in a driver process.

        Example:

            .. testcode::

                import ray

                @ray.remote
                class Actor:
                    def get_task_id(self):
                        return ray.get_runtime_context().get_task_id()

                @ray.remote
                def get_task_id():
                    return ray.get_runtime_context().get_task_id()

                # All the below code generates different task ids.
                a = Actor.remote()
                # Task ids are available for actor tasks.
                print(ray.get(a.get_task_id.remote()))
                # Task ids are available for normal tasks.
                print(ray.get(get_task_id.remote()))

            .. testoutput::
                :options: +MOCK

                16310a0f0a45af5c2746a0e6efb235c0962896a201000000
                c2668a65bda616c1ffffffffffffffffffffffff01000000

        Returns:
            The current worker's task id in hex. None if there's no task id.
        """
    if self.worker.mode != ray._private.worker.WORKER_MODE:
        logger.warning(f'This method is only available when the process is a worker. Current mode: {self.worker.mode}')
        return None
    task_id = self._get_current_task_id()
    return task_id.hex() if not task_id.is_nil() else None