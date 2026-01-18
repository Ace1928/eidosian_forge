import time
from enum import Enum
from typing import Dict, Tuple, Union
from ray.util import PublicAPI
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def disable_ipython():
    """Disable output of IPython HTML objects."""
    try:
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.clear_instance()
    except Exception:
        pass