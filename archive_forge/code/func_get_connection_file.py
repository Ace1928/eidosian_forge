from __future__ import annotations
import json
import sys
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any
import jupyter_client
from jupyter_client import write_connection_file
def get_connection_file(app: IPKernelApp | None=None) -> str:
    """Return the path to the connection file of an app

    Parameters
    ----------
    app : IPKernelApp instance [optional]
        If unspecified, the currently running app will be used
    """
    from traitlets.utils import filefind
    if app is None:
        from ipykernel.kernelapp import IPKernelApp
        if not IPKernelApp.initialized():
            msg = 'app not specified, and not in a running Kernel'
            raise RuntimeError(msg)
        app = IPKernelApp.instance()
    return filefind(app.connection_file, ['.', app.connection_dir])