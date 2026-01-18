from __future__ import annotations
import typing as t
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Request as RequestBase
from werkzeug.wrappers import Response as ResponseBase
from . import json
from .globals import current_app
from .helpers import _split_blueprint_path
@property
def blueprints(self) -> list[str]:
    """The registered names of the current blueprint upwards through
        parent blueprints.

        This will be an empty list if there is no current blueprint, or
        if URL matching failed.

        .. versionadded:: 2.0.1
        """
    name = self.blueprint
    if name is None:
        return []
    return _split_blueprint_path(name)