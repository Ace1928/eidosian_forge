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
def skip_past(self, endtag):
    while self.tokens:
        token = self.next_token()
        if token.token_type == TokenType.BLOCK and token.contents == endtag:
            return
    self.unclosed_block_tag([endtag])