import base64
import hashlib
import hmac
import logging
import random
import string
import sys
import traceback
import zlib
from saml2 import VERSION
from saml2 import saml
from saml2 import samlp
from saml2.time_util import instant
def rec_factory(cls, **kwargs):
    _inst = cls()
    for key, val in kwargs.items():
        if key in ['text', 'lang']:
            setattr(_inst, key, val)
        elif key in _inst.c_attributes:
            try:
                val = str(val)
            except Exception:
                continue
            else:
                setattr(_inst, _inst.c_attributes[key][0], val)
        elif key in _inst.c_child_order:
            for tag, _cls in _inst.c_children.values():
                if tag == key:
                    if isinstance(_cls, list):
                        _cls = _cls[0]
                        claim = []
                        if isinstance(val, list):
                            for v in val:
                                claim.append(rec_factory(_cls, **v))
                        else:
                            claim.append(rec_factory(_cls, **val))
                    else:
                        claim = rec_factory(_cls, **val)
                    setattr(_inst, key, claim)
                    break
    return _inst