import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_with(self) -> nodes.With:
    node = nodes.With(lineno=next(self.stream).lineno)
    targets: t.List[nodes.Expr] = []
    values: t.List[nodes.Expr] = []
    while self.stream.current.type != 'block_end':
        if targets:
            self.stream.expect('comma')
        target = self.parse_assign_target()
        target.set_ctx('param')
        targets.append(target)
        self.stream.expect('assign')
        values.append(self.parse_expression())
    node.targets = targets
    node.values = values
    node.body = self.parse_statements(('name:endwith',), drop_needle=True)
    return node