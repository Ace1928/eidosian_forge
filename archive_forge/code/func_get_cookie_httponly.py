from __future__ import annotations
import hashlib
import typing as t
from collections.abc import MutableMapping
from datetime import datetime
from datetime import timezone
from itsdangerous import BadSignature
from itsdangerous import URLSafeTimedSerializer
from werkzeug.datastructures import CallbackDict
from .json.tag import TaggedJSONSerializer
def get_cookie_httponly(self, app: Flask) -> bool:
    """Returns True if the session cookie should be httponly.  This
        currently just returns the value of the ``SESSION_COOKIE_HTTPONLY``
        config var.
        """
    return app.config['SESSION_COOKIE_HTTPONLY']