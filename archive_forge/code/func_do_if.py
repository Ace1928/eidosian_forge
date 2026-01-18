import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from itertools import cycle as itertools_cycle
from itertools import groupby
from django.conf import settings
from django.utils import timezone
from django.utils.html import conditional_escape, escape, format_html
from django.utils.lorem_ipsum import paragraphs, words
from django.utils.safestring import mark_safe
from .base import (
from .context import Context
from .defaultfilters import date
from .library import Library
from .smartif import IfParser, Literal
@register.tag('if')
def do_if(parser, token):
    """
    Evaluate a variable, and if that variable is "true" (i.e., exists, is not
    empty, and is not a false boolean value), output the contents of the block:

    ::

        {% if athlete_list %}
            Number of athletes: {{ athlete_list|count }}
        {% elif athlete_in_locker_room_list %}
            Athletes should be out of the locker room soon!
        {% else %}
            No athletes.
        {% endif %}

    In the above, if ``athlete_list`` is not empty, the number of athletes will
    be displayed by the ``{{ athlete_list|count }}`` variable.

    The ``if`` tag may take one or several `` {% elif %}`` clauses, as well as
    an ``{% else %}`` clause that will be displayed if all previous conditions
    fail. These clauses are optional.

    ``if`` tags may use ``or``, ``and`` or ``not`` to test a number of
    variables or to negate a given variable::

        {% if not athlete_list %}
            There are no athletes.
        {% endif %}

        {% if athlete_list or coach_list %}
            There are some athletes or some coaches.
        {% endif %}

        {% if athlete_list and coach_list %}
            Both athletes and coaches are available.
        {% endif %}

        {% if not athlete_list or coach_list %}
            There are no athletes, or there are some coaches.
        {% endif %}

        {% if athlete_list and not coach_list %}
            There are some athletes and absolutely no coaches.
        {% endif %}

    Comparison operators are also available, and the use of filters is also
    allowed, for example::

        {% if articles|length >= 5 %}...{% endif %}

    Arguments and operators _must_ have a space between them, so
    ``{% if 1>2 %}`` is not a valid if tag.

    All supported operators are: ``or``, ``and``, ``in``, ``not in``
    ``==``, ``!=``, ``>``, ``>=``, ``<`` and ``<=``.

    Operator precedence follows Python.
    """
    bits = token.split_contents()[1:]
    condition = TemplateIfParser(parser, bits).parse()
    nodelist = parser.parse(('elif', 'else', 'endif'))
    conditions_nodelists = [(condition, nodelist)]
    token = parser.next_token()
    while token.contents.startswith('elif'):
        bits = token.split_contents()[1:]
        condition = TemplateIfParser(parser, bits).parse()
        nodelist = parser.parse(('elif', 'else', 'endif'))
        conditions_nodelists.append((condition, nodelist))
        token = parser.next_token()
    if token.contents == 'else':
        nodelist = parser.parse(('endif',))
        conditions_nodelists.append((None, nodelist))
        token = parser.next_token()
    if token.contents != 'endif':
        raise TemplateSyntaxError('Malformed template tag at line {}: "{}"'.format(token.lineno, token.contents))
    return IfNode(conditions_nodelists)