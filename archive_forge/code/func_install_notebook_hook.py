from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def install_notebook_hook(notebook_type: NotebookType, load: Load, show_doc: ShowDoc, show_app: ShowApp, overwrite: bool=False) -> None:
    """ Install a new notebook display hook.

    Bokeh comes with support for Jupyter notebooks built-in. However, there are
    other kinds of notebooks in use by different communities. This function
    provides a mechanism for other projects to instruct Bokeh how to display
    content in other notebooks.

    This function is primarily of use to developers wishing to integrate Bokeh
    with new notebook types.

    Args:
        notebook_type (str) :
            A name for the notebook type, e.e. ``'Jupyter'`` or ``'Zeppelin'``

            If the name has previously been installed, a ``RuntimeError`` will
            be raised, unless ``overwrite=True``

        load (callable) :
            A function for loading BokehJS in a notebook type. The function
            will be called with the following arguments:

            .. code-block:: python

                load(
                    resources,   # A Resources object for how to load BokehJS
                    verbose,     # Whether to display verbose loading banner
                    hide_banner, # Whether to hide the output banner entirely
                    load_timeout # Time after which to report a load fail error
                )

        show_doc (callable) :
            A function for displaying Bokeh standalone documents in the
            notebook type. This function will be called with the following
            arguments:

            .. code-block:: python

                show_doc(
                    obj,            # the Bokeh object to display
                    state,          # current bokeh.io "state"
                    notebook_handle # whether a notebook handle was requested
                )

            If the notebook platform is capable of supporting in-place updates
            to plots then this function may return an opaque notebook handle
            that can  be used for that purpose. The handle will be returned by
            ``show()``, and can be used by as appropriate to update plots, etc.
            by additional functions in the library that installed the hooks.

        show_app (callable) :
            A function for displaying Bokeh applications in the notebook
            type. This function will be called with the following arguments:

            .. code-block:: python

                show_app(
                    app,          # the Bokeh Application to display
                    state,        # current bokeh.io "state"
                    notebook_url, # URL to the current active notebook page
                    **kw          # any backend-specific keywords passed as-is
                )

        overwrite (bool, optional) :
            Whether to allow an existing hook to be overwritten by a new
            definition (default: False)

    Returns:
        None

    Raises:
        RuntimeError
            If ``notebook_type`` is already installed and ``overwrite=False``

    """
    if notebook_type in _HOOKS and (not overwrite):
        raise RuntimeError(f'hook for notebook type {notebook_type!r} already exists')
    _HOOKS[notebook_type] = Hooks(load=load, doc=show_doc, app=show_app)