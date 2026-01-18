import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def call_release(self, id: bytes) -> None:
    """Attempts to release an object reference.

        When client references are destructed, they release their reference,
        which can opportunistically send a notification through the datachannel
        to release the reference being held for that object on the server.

        Args:
            id: The id of the reference to release on the server side.
        """
    return self.worker.call_release(id)