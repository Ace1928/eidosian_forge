import asyncio
import logging
import types
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, MiddlewareNotUsed
from django.core.signals import request_finished
from django.db import connections, transaction
from django.urls import get_resolver, set_urlconf
from django.utils.log import log_response
from django.utils.module_loading import import_string
from .exception import convert_exception_to_response
def resolve_request(self, request):
    """
        Retrieve/set the urlconf for the request. Return the view resolved,
        with its args and kwargs.
        """
    if hasattr(request, 'urlconf'):
        urlconf = request.urlconf
        set_urlconf(urlconf)
        resolver = get_resolver(urlconf)
    else:
        resolver = get_resolver()
    resolver_match = resolver.resolve(request.path_info)
    request.resolver_match = resolver_match
    return resolver_match