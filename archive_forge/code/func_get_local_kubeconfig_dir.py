import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
def get_local_kubeconfig_dir(env_var: str='KUBECONFIG_DIR', default: str=None) -> str:
    """
    Get the local kubernetes kubeconfig.
    """
    global _kubeconfig_dir
    if _kubeconfig_dir is None:
        _kubeconfig_dir = os.getenv(env_var, default) or os.path.expanduser('~/.kube')
    return _kubeconfig_dir