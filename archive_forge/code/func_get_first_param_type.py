import inspect
import platform
import sys
import threading
from collections.abc import Mapping, Sequence  # noqa: F401
from typing import _GenericAlias
def get_first_param_type(self):
    """
        Return the type annotation of the first argument if it's not empty.
        """
    if not self.sig:
        return None
    params = list(self.sig.parameters.values())
    if params and params[0].annotation is not inspect.Parameter.empty:
        return params[0].annotation
    return None