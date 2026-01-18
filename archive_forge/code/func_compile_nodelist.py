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
def compile_nodelist(self):
    """
        Parse and compile the template source into a nodelist. If debug
        is True and an exception occurs during parsing, the exception is
        annotated with contextual line information where it occurred in the
        template source.
        """
    if self.engine.debug:
        lexer = DebugLexer(self.source)
    else:
        lexer = Lexer(self.source)
    tokens = lexer.tokenize()
    parser = Parser(tokens, self.engine.template_libraries, self.engine.template_builtins, self.origin)
    try:
        return parser.parse()
    except Exception as e:
        if self.engine.debug:
            e.template_debug = self.get_exception_info(e, e.token)
        raise