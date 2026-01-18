from django.template import TemplateSyntaxError
from django.utils.safestring import mark_safe
from django import VERSION as DJANGO_VERSION
from sentry_sdk import _functools, Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
@property
def rendered_content(self):
    hub = Hub.current
    if hub.get_integration(DjangoIntegration) is None:
        return real_rendered_content.fget(self)
    with hub.start_span(op=OP.TEMPLATE_RENDER, description=_get_template_name_description(self.template_name)) as span:
        span.set_data('context', self.context_data)
        return real_rendered_content.fget(self)