import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
@classmethod
def from_params(cls, transport: str, host: str, port: int) -> '_ManagerToken':
    version = cls._version
    pid = os.getpid()
    token = '-'.join([version, str(pid), transport, host, str(port)])
    return cls(token=token)