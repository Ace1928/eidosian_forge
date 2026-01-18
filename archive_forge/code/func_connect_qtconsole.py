from __future__ import annotations
import json
import sys
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any
import jupyter_client
from jupyter_client import write_connection_file
def connect_qtconsole(connection_file: str | None=None, argv: list[str] | None=None) -> Popen[Any]:
    """Connect a qtconsole to the current kernel.

    This is useful for connecting a second qtconsole to a kernel, or to a
    local notebook.

    Parameters
    ----------
    connection_file : str [optional]
        The connection file to be used. Can be given by absolute path, or
        IPython will search in the security directory.
        If run from IPython,

        If unspecified, the connection file for the currently running
        IPython Kernel will be used, which is only allowed from inside a kernel.

    argv : list [optional]
        Any extra args to be passed to the console.

    Returns
    -------
    :class:`subprocess.Popen` instance running the qtconsole frontend
    """
    argv = [] if argv is None else argv
    cf = _find_connection_file(connection_file)
    cmd = ';'.join(['from qtconsole import qtconsoleapp', 'qtconsoleapp.main()'])
    kwargs: dict[str, Any] = {}
    kwargs['start_new_session'] = True
    return Popen([sys.executable, '-c', cmd, '--existing', cf, *argv], stdout=PIPE, stderr=PIPE, close_fds=sys.platform != 'win32', **kwargs)