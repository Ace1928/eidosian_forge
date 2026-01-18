import logging
import os
import sys
import threading
import importlib
import ray
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def set_trace(breakpoint_uuid=None):
    """Interrupt the flow of the program and drop into the Ray debugger.
    Can be used within a Ray task or actor.
    """
    debugpy = _try_import_debugpy()
    if not debugpy:
        return
    _ensure_debugger_port_open_thread_safe()
    _override_breakpoint_hooks()
    with ray._private.worker.global_worker.worker_paused_by_debugger():
        log.info('Waiting for debugger to attach...')
        debugpy.wait_for_client()
    log.info('Debugger client is connected')
    if breakpoint_uuid == POST_MORTEM_ERROR_UUID:
        _debugpy_excepthook()
    else:
        _debugpy_breakpoint()