import importlib.machinery
import logging
import multiprocessing
import os
import queue
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, cast
import wandb
from wandb.sdk.interface.interface import InterfaceBase
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal.internal import wandb_internal
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib.mailbox import Mailbox
from wandb.sdk.wandb_manager import _Manager
from wandb.sdk.wandb_settings import Settings
def ensure_launched(self) -> None:
    """Launch backend worker if not running."""
    if self._manager:
        self._ensure_launched_manager()
        return
    assert self._settings
    settings = self._settings.copy()
    settings.update(_log_level=self._log_level or logging.DEBUG)
    start_method = settings.start_method
    settings_static = SettingsStatic(settings.to_proto())
    user_pid = os.getpid()
    if start_method == 'thread':
        self.record_q = queue.Queue()
        self.result_q = queue.Queue()
        wandb._set_internal_process(disable=True)
        wandb_thread = BackendThread(target=wandb_internal, kwargs=dict(settings=settings_static, record_q=self.record_q, result_q=self.result_q, user_pid=user_pid))
        self.wandb_process = wandb_thread
    else:
        self.record_q = self._multiprocessing.Queue()
        self.result_q = self._multiprocessing.Queue()
        self.wandb_process = self._multiprocessing.Process(target=wandb_internal, kwargs=dict(settings=settings_static, record_q=self.record_q, result_q=self.result_q, user_pid=user_pid))
        assert self.wandb_process
        self.wandb_process.name = 'wandb_internal'
    self._module_main_install()
    logger.info('starting backend process...')
    assert self.wandb_process
    self.wandb_process.start()
    self._internal_pid = self.wandb_process.pid
    logger.info(f'started backend process with pid: {self.wandb_process.pid}')
    self._module_main_uninstall()
    self.interface = InterfaceQueue(process=self.wandb_process, record_q=self.record_q, result_q=self.result_q, mailbox=self._mailbox)