import asyncio
import functools
import os
import re
import signal
import sys
import typing as t
import uuid
import warnings
from asyncio.futures import Future
from concurrent.futures import Future as CFuture
from contextlib import contextmanager
from enum import Enum
import zmq
from jupyter_core.utils import run_sync
from traitlets import (
from traitlets.utils.importstring import import_item
from . import kernelspec
from .asynchronous import AsyncKernelClient
from .blocking import BlockingKernelClient
from .client import KernelClient
from .connect import ConnectionFileMixin
from .managerabc import KernelManagerABC
from .provisioning import KernelProvisionerBase
from .provisioning import KernelProvisionerFactory as KPF  # noqa
def start_new_kernel(startup_timeout: float=60, kernel_name: str='python', **kwargs: t.Any) -> t.Tuple[KernelManager, BlockingKernelClient]:
    """Start a new kernel, and return its Manager and Client"""
    km = KernelManager(kernel_name=kernel_name)
    km.start_kernel(**kwargs)
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=startup_timeout)
    except RuntimeError:
        kc.stop_channels()
        km.shutdown_kernel()
        raise
    return (km, kc)