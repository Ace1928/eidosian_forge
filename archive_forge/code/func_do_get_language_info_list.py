from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
@register.tag('get_language_info_list')
def do_get_language_info_list(parser, token):
    """
    Store a list of language information dictionaries for the given language
    codes in a context variable. The language codes can be specified either as
    a list of strings or a settings.LANGUAGES style list (or any sequence of
    sequences whose first items are language codes).

    Usage::

        {% get_language_info_list for LANGUAGES as langs %}
        {% for l in langs %}
          {{ l.code }}
          {{ l.name }}
          {{ l.name_translated }}
          {{ l.name_local }}
          {{ l.bidi|yesno:"bi-directional,uni-directional" }}
        {% endfor %}
    """
    args = token.split_contents()
    if len(args) != 5 or args[1] != 'for' or args[3] != 'as':
        raise TemplateSyntaxError("'%s' requires 'for sequence as variable' (got %r)" % (args[0], args[1:]))
    return GetLanguageInfoListNode(parser.compile_filter(args[2]), args[4])