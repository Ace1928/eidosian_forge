import inspect
import logging
import re
from enum import Enum
from django.template.context import BaseContext
from django.utils.formats import localize
from django.utils.html import conditional_escape, escape
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import get_text_list, smart_split, unescape_string_literal
from django.utils.timezone import template_localtime
from django.utils.translation import gettext_lazy, pgettext_lazy
from .exceptions import TemplateSyntaxError
def invalid_block_tag(self, token, command, parse_until=None):
    if parse_until:
        raise self.error(token, "Invalid block tag on line %d: '%s', expected %s. Did you forget to register or load this tag?" % (token.lineno, command, get_text_list(["'%s'" % p for p in parse_until], 'or')))
    raise self.error(token, "Invalid block tag on line %d: '%s'. Did you forget to register or load this tag?" % (token.lineno, command))