from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
@register.tag('get_available_languages')
def do_get_available_languages(parser, token):
    """
    Store a list of available languages in the context.

    Usage::

        {% get_available_languages as languages %}
        {% for language in languages %}
        ...
        {% endfor %}

    This puts settings.LANGUAGES into the named variable.
    """
    args = token.contents.split()
    if len(args) != 3 or args[1] != 'as':
        raise TemplateSyntaxError("'get_available_languages' requires 'as variable' (got %r)" % args)
    return GetAvailableLanguagesNode(args[2])