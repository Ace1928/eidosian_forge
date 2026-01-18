import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def get_runtime_context(self):
    """Return a Ray RuntimeContext describing the state on the server

        Returns:
            A RuntimeContext wrapping a client making get_cluster_info calls.
        """
    return _ClientWorkerPropertyAPI(self.worker).build_runtime_context()