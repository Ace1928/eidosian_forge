import importlib.metadata
from typing import Union
from packaging.version import Version, parse
from .constants import STR_OPERATION_TO_FUNC
def is_torch_version(operation: str, version: str):
    """
    Compares the current PyTorch version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(torch_version, operation, version)