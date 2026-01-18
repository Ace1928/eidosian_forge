from __future__ import annotations
import os
import abc
import kr8s
import httpx
import base64
import functools
import contextlib
from pathlib import Path
from lazyops.libs.proxyobj.wraps import proxied
from lazyops.libs.logging import logger
from typing import Optional, Dict, TYPE_CHECKING
from .context import KubernetesContext
def find_kubeconfig_ctx(self, name: str) -> Optional[Path]:
    """
        Find a kubeconfig context
        """
    if self.kconfigs_path:
        for ctx_fname in self.kconfigs_path.iterdir():
            if name in ctx_fname.name:
                logger.info(f'Using kubeconfig: `{ctx_fname.name}`')
                return ctx_fname
    return None