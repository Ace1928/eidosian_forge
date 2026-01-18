from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
@classmethod
def lookup_attr(cls, obj, key):
    __traceback_hide__ = True
    try:
        val = getattr(obj, key)
    except AttributeError:
        if hasattr(obj.__class__, key):
            raise
        else:
            try:
                val = obj[key]
            except (KeyError, TypeError):
                val = cls.undefined(key, owner=obj)
    return val