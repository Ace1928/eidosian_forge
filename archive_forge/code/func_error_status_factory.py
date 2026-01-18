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
def error_status_factory(info):
    if not isinstance(info, Exception):
        status_code_status_code_value, status_message_text = info
    else:
        try:
            exc_val = EXCEPTION2STATUS[info.__class__]
        except KeyError:
            exc_val = samlp.STATUS_AUTHN_FAILED
        try:
            exc_context = info.args[0]
            err_ctx = {'status_message_text': exc_context} if isinstance(exc_context, str) else exc_context
        except IndexError:
            err_ctx = {'status_message_text': str(info)}
        status_message_text = err_ctx.get('status_message_text')
        status_code_status_code_value = err_ctx.get('status_code_status_code_value', exc_val)
    status_msg = samlp.StatusMessage(text=status_message_text) if status_message_text else None
    status = samlp.Status(status_message=status_msg, status_code=samlp.StatusCode(value=samlp.STATUS_RESPONDER, status_code=samlp.StatusCode(value=status_code_status_code_value)))
    return status