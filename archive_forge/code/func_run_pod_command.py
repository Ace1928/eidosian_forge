from __future__ import annotations
import random
import asyncio
import logging
import contextlib
from enum import Enum
from lazyops.libs.kops.base import *
from lazyops.libs.kops.config import KOpsSettings
from lazyops.libs.kops.utils import cached, DillSerializer, SignalHandler
from lazyops.libs.kops._kopf import kopf
from lazyops.types import lazyproperty
from lazyops.utils import logger
from typing import List, Dict, Union, Any, Optional, Callable, TYPE_CHECKING
import lazyops.libs.kops.types as t
import lazyops.libs.kops.atypes as at
def run_pod_command(name: str, namespace: str, command: Union[str, List[str]], container: Optional[str]=None, stderr: Optional[bool]=True, stdin: Optional[bool]=True, stdout: Optional[bool]=True, tty: Optional[bool]=False, ignore_error: bool=True, **kwargs) -> str:
    """
    Runs a command in the pod and returns the command
    name = 'name_example' # str | name of the PodExecOptions
    namespace = 'namespace_example' # str | object name and auth scope, such as for teams and projects
    command = 'command_example' # str | Command is the remote command to execute. argv array. Not executed within a shell. (optional)
    container = 'container_example' # str | Container in which to execute the command. Defaults to only container if there is only one container in the pod. (optional)
    stderr = True # bool | Redirect the standard error stream of the pod for this call. (optional)
    stdin = True # bool | Redirect the standard input stream of the pod for this call. Defaults to false. (optional)
    stdout = True # bool | Redirect the standard output stream of the pod for this call. (optional)
    tty = True # bool | TTY if true indicates that a tty will be allocated for the exec call. Defaults to false. (optional)

    """
    if isinstance(command, str):
        command = command.split(' ')
    try:
        return BaseKOpsClient.core_v1_ws.connect_post_namespaced_pod_exec(name=name, namespace=namespace, command=command, container=container, stderr=stderr, stdin=stdin, stdout=stdout, tty=tty)
    except Exception as e:
        if not ignore_error:
            logger.warning(f'[{namespace}/{name}] Exec Status Failed: {e}')
        return None