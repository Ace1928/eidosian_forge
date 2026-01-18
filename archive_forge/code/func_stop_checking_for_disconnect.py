import asyncio
import logging
import time
from abc import ABC, abstractmethod
from asyncio.tasks import FIRST_COMPLETED
from typing import Any, Callable, Optional, Union
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import calculate_remaining_timeout
from ray.serve.handle import DeploymentResponse, DeploymentResponseGenerator
def stop_checking_for_disconnect(self):
    """Once this is called, the disconnected_task will be ignored."""
    self._disconnected_task = None