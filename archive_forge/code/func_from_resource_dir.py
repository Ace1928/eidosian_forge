from __future__ import annotations
import json
import os
import re
import shutil
import typing as t
import warnings
from jupyter_core.paths import SYSTEM_JUPYTER_PATH, jupyter_data_dir, jupyter_path
from traitlets import Bool, CaselessStrEnum, Dict, HasTraits, List, Set, Type, Unicode, observe
from traitlets.config import LoggingConfigurable
from .provisioning import KernelProvisionerFactory as KPF  # noqa
@classmethod
def from_resource_dir(cls: type[KernelSpec], resource_dir: str) -> KernelSpec:
    """Create a KernelSpec object by reading kernel.json

        Pass the path to the *directory* containing kernel.json.
        """
    kernel_file = pjoin(resource_dir, 'kernel.json')
    with open(kernel_file, encoding='utf-8') as f:
        kernel_dict = json.load(f)
    return cls(resource_dir=resource_dir, **kernel_dict)