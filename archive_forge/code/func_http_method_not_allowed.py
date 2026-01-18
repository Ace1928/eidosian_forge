import logging
from asgiref.sync import iscoroutinefunction, markcoroutinefunction
from django.core.exceptions import ImproperlyConfigured
from django.http import (
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils.decorators import classonlymethod
from django.utils.functional import classproperty
def http_method_not_allowed(self, request, *args, **kwargs):
    logger.warning('Method Not Allowed (%s): %s', request.method, request.path, extra={'status_code': 405, 'request': request})
    response = HttpResponseNotAllowed(self._allowed_methods())
    if self.view_is_async:

        async def func():
            return response
        return func()
    else:
        return response