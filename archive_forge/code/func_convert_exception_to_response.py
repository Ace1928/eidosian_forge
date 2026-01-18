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
def convert_exception_to_response(get_response):
    """
    Wrap the given get_response callable in exception-to-response conversion.

    All exceptions will be converted. All known 4xx exceptions (Http404,
    PermissionDenied, MultiPartParserError, SuspiciousOperation) will be
    converted to the appropriate response, and all other exceptions will be
    converted to 500 responses.

    This decorator is automatically applied to all middleware to ensure that
    no middleware leaks an exception and that the next middleware in the stack
    can rely on getting a response instead of an exception.
    """
    if iscoroutinefunction(get_response):

        @wraps(get_response)
        async def inner(request):
            try:
                response = await get_response(request)
            except Exception as exc:
                response = await sync_to_async(response_for_exception, thread_sensitive=False)(request, exc)
            return response
        return inner
    else:

        @wraps(get_response)
        def inner(request):
            try:
                response = get_response(request)
            except Exception as exc:
                response = response_for_exception(request, exc)
            return response
        return inner