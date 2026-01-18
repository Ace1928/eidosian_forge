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
def push_notebook(*, document: Document | None=None, state: State | None=None, handle: CommsHandle | None=None) -> None:
    """ Update Bokeh plots in a Jupyter notebook output cells with new data
    or property values.

    When working the the notebook, the ``show`` function can be passed the
    argument ``notebook_handle=True``, which will cause it to return a
    handle object that can be used to update the Bokeh output later. When
    ``push_notebook`` is called, any property updates (e.g. plot titles or
    data source values, etc.) since the last call to ``push_notebook`` or
    the original ``show`` call are applied to the Bokeh output in the
    previously rendered Jupyter output cell.

    Several example notebooks can be found in the GitHub repository in
    the :bokeh-tree:`examples/output/jupyter/push_notebook` directory.

    Args:

        document (Document, optional):
            A |Document| to push from. If None uses ``curdoc()``. (default:
            None)

        state (State, optional) :
            A :class:`State` object. If None, then the current default
            state (set by |output_file|, etc.) is used. (default: None)

    Returns:
        None

    Examples:

        Typical usage is typically similar to this:

        .. code-block:: python

            from bokeh.plotting import figure
            from bokeh.io import output_notebook, push_notebook, show

            output_notebook()

            plot = figure()
            plot.circle([1,2,3], [4,6,5])

            handle = show(plot, notebook_handle=True)

            # Update the plot title in the earlier cell
            plot.title.text = "New Title"
            push_notebook(handle=handle)

    """
    from ..protocol import Protocol as BokehProtocol
    if state is None:
        state = curstate()
    if not document:
        document = state.document
    if not document:
        warn('No document to push')
        return
    if handle is None:
        handle = state.last_comms_handle
    if not handle:
        warn('Cannot find a last shown plot to update. Call output_notebook() and show(..., notebook_handle=True) before push_notebook()')
        return
    events = list(handle.doc.callbacks._held_events)
    if len(events) == 0:
        return
    handle.doc.callbacks._held_events = []
    msg = BokehProtocol().create('PATCH-DOC', cast(list['DocumentPatchedEvent'], events))
    handle.comms.send(msg.header_json)
    handle.comms.send(msg.metadata_json)
    handle.comms.send(msg.content_json)
    for buffer in msg.buffers:
        header = json.dumps(buffer.ref)
        payload = buffer.to_bytes()
        handle.comms.send(header)
        handle.comms.send(buffers=[payload])