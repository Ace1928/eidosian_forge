from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
@register.tag('get_language_info')
def do_get_language_info(parser, token):
    """
    Store the language information dictionary for the given language code in a
    context variable.

    Usage::

        {% get_language_info for LANGUAGE_CODE as l %}
        {{ l.code }}
        {{ l.name }}
        {{ l.name_translated }}
        {{ l.name_local }}
        {{ l.bidi|yesno:"bi-directional,uni-directional" }}
    """
    args = token.split_contents()
    if len(args) != 5 or args[1] != 'for' or args[3] != 'as':
        raise TemplateSyntaxError("'%s' requires 'for string as variable' (got %r)" % (args[0], args[1:]))
    return GetLanguageInfoNode(parser.compile_filter(args[2]), args[4])