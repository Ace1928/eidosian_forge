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
def handle_uncaught_exception(self, request, resolver, exc_info):
    """Last-chance handler for exceptions."""
    try:
        return super().handle_uncaught_exception(request, resolver, exc_info)
    except Exception:
        return HttpResponseServerError(traceback.format_exc() if settings.DEBUG else 'Internal Server Error', content_type='text/plain')