import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
def is_provisioner_available(self, kernel_spec: Any) -> bool:
    """
        Reads the associated ``kernel_spec`` to determine the provisioner and returns whether it
        exists as an entry_point (True) or not (False).  If the referenced provisioner is not
        in the current cache or cannot be loaded via entry_points, a warning message is issued
        indicating it is not available.
        """
    is_available: bool = True
    provisioner_cfg = self._get_provisioner_config(kernel_spec)
    provisioner_name = str(provisioner_cfg.get('provisioner_name'))
    if not self._check_availability(provisioner_name):
        is_available = False
        self.log.warning(f"Kernel '{kernel_spec.display_name}' is referencing a kernel provisioner ('{provisioner_name}') that is not available.  Ensure the appropriate package has been installed and retry.")
    return is_available