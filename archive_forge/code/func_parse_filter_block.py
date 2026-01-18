import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_filter_block(self) -> nodes.FilterBlock:
    node = nodes.FilterBlock(lineno=next(self.stream).lineno)
    node.filter = self.parse_filter(None, start_inline=True)
    node.body = self.parse_statements(('name:endfilter',), drop_needle=True)
    return node