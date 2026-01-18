import logging
import sys
from functools import wraps
from asgiref.sync import iscoroutinefunction, sync_to_async
from django.conf import settings
from django.core import signals
from django.core.exceptions import (
from django.http import Http404
from django.http.multipartparser import MultiPartParserError
from django.urls import get_resolver, get_urlconf
from django.utils.log import log_response
from django.views import debug
def get_exception_response(request, resolver, status_code, exception):
    try:
        callback = resolver.resolve_error_handler(status_code)
        response = callback(request, exception=exception)
    except Exception:
        signals.got_request_exception.send(sender=None, request=request)
        response = handle_uncaught_exception(request, resolver, sys.exc_info())
    return response