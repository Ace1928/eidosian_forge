import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union
from traitlets.config import Instance, LoggingConfigurable, Unicode
from ..connect import KernelConnectionInfo
def get_shutdown_wait_time(self, recommended: float=5.0) -> float:
    """
        Returns the time allowed for a complete shutdown. This may vary by provisioner.

        This method is called from `KernelManager.finish_shutdown()` during the graceful
        phase of its kernel shutdown sequence.

        The recommended value will typically be what is configured in the kernel manager.
        """
    return recommended