from django.template import TemplateSyntaxError
from django.utils.safestring import mark_safe
from django import VERSION as DJANGO_VERSION
from sentry_sdk import _functools, Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
def get_template_frame_from_exception(exc_value):
    if hasattr(exc_value, 'template_debug'):
        return _get_template_frame_from_debug(exc_value.template_debug)
    if hasattr(exc_value, 'django_template_source'):
        return _get_template_frame_from_source(exc_value.django_template_source)
    if isinstance(exc_value, TemplateSyntaxError) and hasattr(exc_value, 'source'):
        source = exc_value.source
        if isinstance(source, (tuple, list)) and isinstance(source[0], Origin):
            return _get_template_frame_from_source(source)
    return None