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
def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
    """Returns a :class:`KernelSpec` instance for the given kernel_name.

        Raises :exc:`NoSuchKernel` if the given kernel name is not found.
        """
    if not _is_valid_kernel_name(kernel_name):
        self.log.warning(f'Kernelspec name {kernel_name} is invalid: {_kernel_name_description}')
    resource_dir = self._find_spec_directory(kernel_name.lower())
    if resource_dir is None:
        self.log.warning('Kernelspec name %s cannot be found!', kernel_name)
        raise NoSuchKernel(kernel_name)
    return self._get_kernel_spec_by_name(kernel_name, resource_dir)