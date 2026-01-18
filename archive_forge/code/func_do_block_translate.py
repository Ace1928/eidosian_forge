from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
@register.tag('blocktranslate')
@register.tag('blocktrans')
def do_block_translate(parser, token):
    """
    Translate a block of text with parameters.

    Usage::

        {% blocktranslate with bar=foo|filter boo=baz|filter %}
        This is {{ bar }} and {{ boo }}.
        {% endblocktranslate %}

    Additionally, this supports pluralization::

        {% blocktranslate count count=var|length %}
        There is {{ count }} object.
        {% plural %}
        There are {{ count }} objects.
        {% endblocktranslate %}

    This is much like ngettext, only in template syntax.

    The "var as value" legacy format is still supported::

        {% blocktranslate with foo|filter as bar and baz|filter as boo %}
        {% blocktranslate count var|length as count %}

    The translated string can be stored in a variable using `asvar`::

        {% blocktranslate with bar=foo|filter boo=baz|filter asvar var %}
        This is {{ bar }} and {{ boo }}.
        {% endblocktranslate %}
        {{ var }}

    Contextual translations are supported::

        {% blocktranslate with bar=foo|filter context "greeting" %}
            This is {{ bar }}.
        {% endblocktranslate %}

    This is equivalent to calling pgettext/npgettext instead of
    (u)gettext/(u)ngettext.
    """
    bits = token.split_contents()
    options = {}
    remaining_bits = bits[1:]
    asvar = None
    while remaining_bits:
        option = remaining_bits.pop(0)
        if option in options:
            raise TemplateSyntaxError('The %r option was specified more than once.' % option)
        if option == 'with':
            value = token_kwargs(remaining_bits, parser, support_legacy=True)
            if not value:
                raise TemplateSyntaxError('"with" in %r tag needs at least one keyword argument.' % bits[0])
        elif option == 'count':
            value = token_kwargs(remaining_bits, parser, support_legacy=True)
            if len(value) != 1:
                raise TemplateSyntaxError('"count" in %r tag expected exactly one keyword argument.' % bits[0])
        elif option == 'context':
            try:
                value = remaining_bits.pop(0)
                value = parser.compile_filter(value)
            except Exception:
                raise TemplateSyntaxError('"context" in %r tag expected exactly one argument.' % bits[0])
        elif option == 'trimmed':
            value = True
        elif option == 'asvar':
            try:
                value = remaining_bits.pop(0)
            except IndexError:
                raise TemplateSyntaxError("No argument provided to the '%s' tag for the asvar option." % bits[0])
            asvar = value
        else:
            raise TemplateSyntaxError('Unknown argument for %r tag: %r.' % (bits[0], option))
        options[option] = value
    if 'count' in options:
        countervar, counter = next(iter(options['count'].items()))
    else:
        countervar, counter = (None, None)
    if 'context' in options:
        message_context = options['context']
    else:
        message_context = None
    extra_context = options.get('with', {})
    trimmed = options.get('trimmed', False)
    singular = []
    plural = []
    while parser.tokens:
        token = parser.next_token()
        if token.token_type in (TokenType.VAR, TokenType.TEXT):
            singular.append(token)
        else:
            break
    if countervar and counter:
        if token.contents.strip() != 'plural':
            raise TemplateSyntaxError("%r doesn't allow other block tags inside it" % bits[0])
        while parser.tokens:
            token = parser.next_token()
            if token.token_type in (TokenType.VAR, TokenType.TEXT):
                plural.append(token)
            else:
                break
    end_tag_name = 'end%s' % bits[0]
    if token.contents.strip() != end_tag_name:
        raise TemplateSyntaxError("%r doesn't allow other block tags (seen %r) inside it" % (bits[0], token.contents))
    return BlockTranslateNode(extra_context, singular, plural, countervar, counter, message_context, trimmed=trimmed, asvar=asvar, tag_name=bits[0])