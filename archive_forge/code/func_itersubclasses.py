import base64
import hashlib
import hmac
import json
import os
import uuid
from oslo_utils import secretutils
from oslo_utils import uuidutils
def itersubclasses(cls, _seen=None):
    """Generator over all subclasses of a given class in depth first order."""
    _seen = _seen or set()
    try:
        subs = cls.__subclasses__()
    except TypeError:
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub