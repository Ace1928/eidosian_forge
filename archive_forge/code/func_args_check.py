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
def args_check(name, func, provided):
    provided = list(provided)
    plen = len(provided) + 1
    func = inspect.unwrap(func)
    args, _, _, defaults, _, _, _ = inspect.getfullargspec(func)
    alen = len(args)
    dlen = len(defaults or [])
    if plen < alen - dlen or plen > alen:
        raise TemplateSyntaxError('%s requires %d arguments, %d provided' % (name, alen - dlen, plen))
    return True