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
def extend_nodelist(self, nodelist, node, token):
    if node.must_be_first and nodelist.contains_nontext:
        raise self.error(token, '%r must be the first tag in the template.' % node)
    if not isinstance(node, TextNode):
        nodelist.contains_nontext = True
    node.token = token
    node.origin = self.origin
    nodelist.append(node)