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
def get_all_specs(self) -> dict[str, t.Any]:
    """Returns a dict mapping kernel names to kernelspecs.

        Returns a dict of the form::

            {
              'kernel_name': {
                'resource_dir': '/path/to/kernel_name',
                'spec': {"the spec itself": ...}
              },
              ...
            }
        """
    d = self.find_kernel_specs()
    res = {}
    for kname, resource_dir in d.items():
        try:
            if self.__class__ is KernelSpecManager:
                spec = self._get_kernel_spec_by_name(kname, resource_dir)
            else:
                spec = self.get_kernel_spec(kname)
            res[kname] = {'resource_dir': resource_dir, 'spec': spec.to_dict()}
        except NoSuchKernel:
            pass
        except Exception:
            self.log.warning('Error loading kernelspec %r', kname, exc_info=True)
    return res