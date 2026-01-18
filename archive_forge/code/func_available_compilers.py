from typing import List, Optional
from sys import version_info
from importlib import reload, metadata
from collections import defaultdict
import dataclasses
import re
from semantic_version import Version
def available_compilers() -> List[str]:
    """Load and return a list of available compilers that are
    installed and compatible with the :func:`~.qjit` decorator.

    **Example**

    This method returns the name of installed compiler packages supported in
    PennyLane. For example, after installing the
    `Catalyst <https://github.com/pennylaneai/catalyst>`__
    compiler, this will now appear as an available compiler:

    >>> qml.compiler.available_compilers()
    ['catalyst']
    """
    _reload_compilers()
    return list(AvailableCompilers.names_entrypoints.keys())