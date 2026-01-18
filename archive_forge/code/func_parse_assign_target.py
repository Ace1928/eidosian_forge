import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def parse_assign_target(self, with_tuple: bool=True, name_only: bool=False, extra_end_rules: t.Optional[t.Tuple[str, ...]]=None, with_namespace: bool=False) -> t.Union[nodes.NSRef, nodes.Name, nodes.Tuple]:
    """Parse an assignment target.  As Jinja allows assignments to
        tuples, this function can parse all allowed assignment targets.  Per
        default assignments to tuples are parsed, that can be disable however
        by setting `with_tuple` to `False`.  If only assignments to names are
        wanted `name_only` can be set to `True`.  The `extra_end_rules`
        parameter is forwarded to the tuple parsing function.  If
        `with_namespace` is enabled, a namespace assignment may be parsed.
        """
    target: nodes.Expr
    if with_namespace and self.stream.look().type == 'dot':
        token = self.stream.expect('name')
        next(self.stream)
        attr = self.stream.expect('name')
        target = nodes.NSRef(token.value, attr.value, lineno=token.lineno)
    elif name_only:
        token = self.stream.expect('name')
        target = nodes.Name(token.value, 'store', lineno=token.lineno)
    else:
        if with_tuple:
            target = self.parse_tuple(simplified=True, extra_end_rules=extra_end_rules)
        else:
            target = self.parse_primary()
        target.set_ctx('store')
    if not target.can_assign():
        self.fail(f"can't assign to {type(target).__name__.lower()!r}", target.lineno)
    return target