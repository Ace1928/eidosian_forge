import pprint
import re
import typing as t
from markupsafe import Markup
from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context
def find_comments(self, lineno: int) -> t.List[str]:
    if not self.comment_tags or self.last_lineno > lineno:
        return []
    for idx, (token_lineno, _, _) in enumerate(self.tokens[self.offset:]):
        if token_lineno > lineno:
            return self.find_backwards(self.offset + idx)
    return self.find_backwards(len(self.tokens))