from typing import List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
from .parse import ParseError, parse
def attrs_block_plugin(md: MarkdownIt) -> None:
    """Parse block attributes.

    Block attributes are attributes on a single line, with no other content.
    They attach the specified attributes to the block below them::

        {.a #b c=1}
        A paragraph, that will be assigned the class ``a`` and the identifier ``b``.

    Attributes can be stacked, with classes accumulating and lower attributes overriding higher::

        {#a .a c=1}
        {#b .b c=2}
        A paragraph, that will be assigned the class ``a b c``, and the identifier ``b``.

    This syntax is inspired by Djot block attributes.
    """
    md.block.ruler.before('fence', 'attr', _attr_block_rule)
    md.core.ruler.after('block', 'attr', _attr_resolve_block_rule)