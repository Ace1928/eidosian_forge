from __future__ import annotations
import asyncio
import json
import os
import socket
import typing as t
import uuid
from functools import wraps
from pathlib import Path
import zmq
from traitlets import Any, Bool, Dict, DottedObjectName, Instance, Unicode, default, observe
from traitlets.config.configurable import LoggingConfigurable
from traitlets.utils.importstring import import_item
from .connect import KernelConnectionInfo
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
from .utils import ensure_async, run_sync, utcnow
def list_kernel_ids(self) -> list[str]:
    """Return a list of the kernel ids of the active kernels."""
    if self.external_connection_dir is not None:
        external_connection_dir = Path(self.external_connection_dir)
        if external_connection_dir.is_dir():
            connection_files = [p for p in external_connection_dir.iterdir() if p.is_file()]
            k = list(self.kernel_id_to_connection_file.keys())
            v = list(self.kernel_id_to_connection_file.values())
            for connection_file in list(self.kernel_id_to_connection_file.values()):
                if connection_file not in connection_files:
                    kernel_id = k[v.index(connection_file)]
                    del self.kernel_id_to_connection_file[kernel_id]
                    del self._kernels[kernel_id]
            for connection_file in connection_files:
                if connection_file in self.kernel_id_to_connection_file.values():
                    continue
                try:
                    connection_info: KernelConnectionInfo = json.loads(connection_file.read_text())
                except Exception:
                    continue
                self.log.debug('Loading connection file %s', connection_file)
                if not ('kernel_name' in connection_info and 'key' in connection_info):
                    continue
                kernel_id = self.new_kernel_id()
                self.kernel_id_to_connection_file[kernel_id] = connection_file
                km = self.kernel_manager_factory(parent=self, log=self.log, owns_kernel=False)
                km.load_connection_info(connection_info)
                km.last_activity = utcnow()
                km.execution_state = 'idle'
                km.connections = 1
                km.kernel_id = kernel_id
                km.kernel_name = connection_info['kernel_name']
                km.ready.set_result(None)
                self._kernels[kernel_id] = km
    return list(self._kernels.keys())