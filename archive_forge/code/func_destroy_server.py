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
def destroy_server(server_id: ID) -> None:
    """ Given a UUID id of a div removed or replaced in the Jupyter
    notebook, destroy the corresponding server sessions and stop it.

    """
    server = curstate().uuid_to_server.get(server_id, None)
    if server is None:
        log.debug(f'No server instance found for uuid: {server_id!r}')
        return
    try:
        for session in server.get_sessions():
            session.destroy()
        server.stop()
        del curstate().uuid_to_server[server_id]
    except Exception as e:
        log.debug(f'Could not destroy server for id {server_id!r}: {e}')