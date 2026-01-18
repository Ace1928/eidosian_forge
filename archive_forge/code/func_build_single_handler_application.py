from __future__ import annotations
import logging # isort:skip
import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Iterator
from bokeh.application import Application
from bokeh.application.handlers import (
from bokeh.util.warnings import warn
def build_single_handler_application(path: str, argv: list[str] | None=None) -> Application:
    """ Return a Bokeh application built using a single handler for a script,
    notebook, or directory.

    In general a Bokeh :class:`~bokeh.application.application.Application` may
    have any number of handlers to initialize |Document| objects for new client
    sessions. However, in many cases only a single handler is needed. This
    function examines the ``path`` provided, and returns an ``Application``
    initialized with one of the following handlers:

    * :class:`~bokeh.application.handlers.script.ScriptHandler` when ``path``
      is to a ``.py`` script.

    * :class:`~bokeh.application.handlers.notebook.NotebookHandler` when
      ``path`` is to an ``.ipynb`` Jupyter notebook.

    * :class:`~bokeh.application.handlers.directory.DirectoryHandler` when
      ``path`` is to a directory containing a ``main.py`` script.

    Args:
        path (str) : path to a file or directory for creating a Bokeh
            application.

        argv (seq[str], optional) : command line arguments to pass to the
            application handler

    Returns:
        :class:`~bokeh.application.application.Application`

    Raises:
        RuntimeError

    Notes:
        If ``path`` ends with a file ``main.py`` then a warning will be printed
        regarding running directory-style apps by passing the directory instead.

    """
    argv = argv or []
    path = os.path.abspath(os.path.expanduser(path))
    handler: Handler
    if os.path.isdir(path):
        handler = DirectoryHandler(filename=path, argv=argv)
    elif os.path.isfile(path):
        if path.endswith('.ipynb'):
            handler = NotebookHandler(filename=path, argv=argv)
        elif path.endswith('.py'):
            if path.endswith('main.py'):
                warn(DIRSTYLE_MAIN_WARNING)
            handler = ScriptHandler(filename=path, argv=argv)
        else:
            raise ValueError("Expected a '.py' script or '.ipynb' notebook, got: '%s'" % path)
    else:
        raise ValueError('Path for Bokeh server application does not exist: %s' % path)
    if handler.failed:
        raise RuntimeError(f'Error loading {path}:\n\n{handler.error}\n{handler.error_detail} ')
    application = Application(handler)
    return application