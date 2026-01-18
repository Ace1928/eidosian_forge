import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union
from traitlets.config import Instance, LoggingConfigurable, Unicode
from ..connect import KernelConnectionInfo
def get_stable_start_time(self, recommended: float=10.0) -> float:
    """
        Returns the expected upper bound for a kernel (re-)start to complete.
        This may vary by provisioner.

        The recommended value will typically be what is configured in the kernel restarter.
        """
    return recommended