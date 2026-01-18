import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
def get_local(self, name: str, *, stacklevel=0) -> ComptimeVar:
    """
        Retrieve the compile-time known information about a local.
        """
    tx = self.__get_tx(stacklevel)
    return ComptimeVar(tx.symbolic_locals[name])