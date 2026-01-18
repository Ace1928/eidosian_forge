from __future__ import annotations
import os
import abc
import atexit
import pathlib
import filelock
import contextlib
from lazyops.types import BaseModel, Field
from lazyops.utils.logs import logger
from lazyops.utils.serialization import Json
from typing import Optional, Dict, Any, Set, List, Union, Generator, TYPE_CHECKING
def get_kubeconfig(self, name: Optional[str]=None, set_as_envval: Optional[bool]=True, set_active: Optional[bool]=False) -> str:
    """
        Returns the kubeconfig for the given context
        """
    name = name or self.k8s_active_ctx
    if name is not None and name in self.k8s_kubeconfigs:
        kconfig = self.k8s_kubeconfigs[name]
    else:
        from lazyops.utils.system import get_local_kubeconfig
        kconfig = get_local_kubeconfig(name=name, set_as_envval=False)
        if not name:
            name = pathlib.Path(kconfig).stem
        self.k8s_kubeconfigs[name] = kconfig
    if set_as_envval:
        os.environ['KUBECONFIG'] = kconfig
    if not self.k8s_active_ctx or set_active:
        self.k8s_active_ctx = name
    return kconfig