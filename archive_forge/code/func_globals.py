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
def globals(cls, data):
    """Construct the globals dictionary to use as the execution context for
        the expression or suite.
        """
    return {'__data__': data, '_lookup_name': cls.lookup_name, '_lookup_attr': cls.lookup_attr, '_lookup_item': cls.lookup_item, 'UndefinedError': UndefinedError}