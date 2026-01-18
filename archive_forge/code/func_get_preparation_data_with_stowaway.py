from __future__ import annotations
import multiprocessing
import multiprocessing.process
import os
import os.path
import sys
import traceback
from typing import Any
from coverage.debug import DebugControl
def get_preparation_data_with_stowaway(name: str) -> dict[str, Any]:
    """Get the original preparation data, and also insert our stowaway."""
    d = original_get_preparation_data(name)
    d['stowaway'] = Stowaway(rcfile)
    return d