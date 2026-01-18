from urllib.parse import quote, urljoin
from django import template
from django.apps import apps
from django.utils.encoding import iri_to_uri
from django.utils.html import conditional_escape
@register.tag
def get_static_prefix(parser, token):
    """
    Populate a template variable with the static prefix,
    ``settings.STATIC_URL``.

    Usage::

        {% get_static_prefix [as varname] %}

    Examples::

        {% get_static_prefix %}
        {% get_static_prefix as static_prefix %}
    """
    return PrefixNode.handle_token(parser, token, 'STATIC_URL')