import logging
import os
import re
import signal
import sys
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, List, Set, Union
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS, _PYTHON_GREATER_EQUAL_3_8_0
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_info
def register_signal_handlers(self) -> None:
    self.received_sigterm = False
    self._original_handlers = self._get_current_signal_handlers()
    sigusr_handlers: List[_HANDLER] = []
    sigterm_handlers: List[_HANDLER] = [self._sigterm_notifier_fn]
    environment = self.trainer._accelerator_connector.cluster_environment
    if isinstance(environment, SLURMEnvironment) and environment.auto_requeue:
        log.info('SLURM auto-requeueing enabled. Setting signal handlers.')
        sigusr_handlers.append(self._slurm_sigusr_handler_fn)
        sigterm_handlers.append(self._sigterm_handler_fn)
    if not self._is_on_windows():
        sigusr = environment.requeue_signal if isinstance(environment, SLURMEnvironment) else signal.SIGUSR1
        assert sigusr is not None
        if sigusr_handlers and (not self._has_already_handler(sigusr)):
            self._register_signal(sigusr, _HandlersCompose(sigusr_handlers))
        if self._has_already_handler(signal.SIGTERM):
            sigterm_handlers.append(signal.getsignal(signal.SIGTERM))
        self._register_signal(signal.SIGTERM, _HandlersCompose(sigterm_handlers))