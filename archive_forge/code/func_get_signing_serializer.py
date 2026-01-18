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
def get_signing_serializer(self, app: Flask) -> URLSafeTimedSerializer | None:
    if not app.secret_key:
        return None
    signer_kwargs = dict(key_derivation=self.key_derivation, digest_method=self.digest_method)
    return URLSafeTimedSerializer(app.secret_key, salt=self.salt, serializer=self.serializer, signer_kwargs=signer_kwargs)