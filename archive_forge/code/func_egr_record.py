import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
@property
def egr_record(self) -> ExecutableGroupResultFilesystemRecord:
    """The `cg.ExecutablegroupResultFilesystemRecord` keeping track of all the paths for saved
        files."""
    return self._egr_record