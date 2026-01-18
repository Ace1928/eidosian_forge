import asyncio
import logging
import sys
import tempfile
import traceback
from contextlib import aclosing
from asgiref.sync import ThreadSensitiveContext, sync_to_async
from django.conf import settings
from django.core import signals
from django.core.exceptions import RequestAborted, RequestDataTooBig
from django.core.handlers import base
from django.http import (
from django.urls import set_script_prefix
from django.utils.functional import cached_property
def create_request(self, scope, body_file):
    """
        Create the Request object and returns either (request, None) or
        (None, response) if there is an error response.
        """
    try:
        return (self.request_class(scope, body_file), None)
    except UnicodeDecodeError:
        logger.warning('Bad Request (UnicodeDecodeError)', exc_info=sys.exc_info(), extra={'status_code': 400})
        return (None, HttpResponseBadRequest())
    except RequestDataTooBig:
        return (None, HttpResponse('413 Payload too large', status=413))