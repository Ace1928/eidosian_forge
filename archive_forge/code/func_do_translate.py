from decimal import Decimal
from django.conf import settings
from django.template import Library, Node, TemplateSyntaxError, Variable
from django.template.base import TokenType, render_value_in_context
from django.template.defaulttags import token_kwargs
from django.utils import translation
from django.utils.safestring import SafeData, SafeString, mark_safe
@register.tag('translate')
@register.tag('trans')
def do_translate(parser, token):
    """
    Mark a string for translation and translate the string for the current
    language.

    Usage::

        {% translate "this is a test" %}

    This marks the string for translation so it will be pulled out by
    makemessages into the .po files and runs the string through the translation
    engine.

    There is a second form::

        {% translate "this is a test" noop %}

    This marks the string for translation, but returns the string unchanged.
    Use it when you need to store values into forms that should be translated
    later on.

    You can use variables instead of constant strings
    to translate stuff you marked somewhere else::

        {% translate variable %}

    This tries to translate the contents of the variable ``variable``. Make
    sure that the string in there is something that is in the .po file.

    It is possible to store the translated string into a variable::

        {% translate "this is a test" as var %}
        {{ var }}

    Contextual translations are also supported::

        {% translate "this is a test" context "greeting" %}

    This is equivalent to calling pgettext instead of (u)gettext.
    """
    bits = token.split_contents()
    if len(bits) < 2:
        raise TemplateSyntaxError("'%s' takes at least one argument" % bits[0])
    message_string = parser.compile_filter(bits[1])
    remaining = bits[2:]
    noop = False
    asvar = None
    message_context = None
    seen = set()
    invalid_context = {'as', 'noop'}
    while remaining:
        option = remaining.pop(0)
        if option in seen:
            raise TemplateSyntaxError("The '%s' option was specified more than once." % option)
        elif option == 'noop':
            noop = True
        elif option == 'context':
            try:
                value = remaining.pop(0)
            except IndexError:
                raise TemplateSyntaxError("No argument provided to the '%s' tag for the context option." % bits[0])
            if value in invalid_context:
                raise TemplateSyntaxError("Invalid argument '%s' provided to the '%s' tag for the context option" % (value, bits[0]))
            message_context = parser.compile_filter(value)
        elif option == 'as':
            try:
                value = remaining.pop(0)
            except IndexError:
                raise TemplateSyntaxError("No argument provided to the '%s' tag for the as option." % bits[0])
            asvar = value
        else:
            raise TemplateSyntaxError('Unknown argument for \'%s\' tag: \'%s\'. The only options available are \'noop\', \'context\' "xxx", and \'as VAR\'.' % (bits[0], option))
        seen.add(option)
    return TranslateNode(message_string, noop, asvar, message_context)