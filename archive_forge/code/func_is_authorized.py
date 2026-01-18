from __future__ import annotations
from typing import TYPE_CHECKING, Awaitable
from traitlets import Instance
from traitlets.config import LoggingConfigurable
from .identity import IdentityProvider, User
def is_authorized(self, handler: JupyterHandler, user: User, action: str, resource: str) -> bool:
    """This method always returns True.

        All authenticated users are allowed to do anything in the Jupyter Server.
        """
    return True