from contextlib import contextmanager
from typing import Iterator, Optional, Tuple
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def fieldlist_plugin(md: MarkdownIt) -> None:
    """Field lists are mappings from field names to field bodies, based on the
    `reStructureText syntax
    <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#field-lists>`_.

    .. code-block:: md

        :name *markup*:
        :name1: body content
        :name2: paragraph 1

                paragraph 2
        :name3:
          paragraph 1

          paragraph 2

    A field name may consist of any characters except colons (":").
    Inline markup is parsed in field names.

    The field name is followed by whitespace and the field body.
    The field body may be empty or contain multiple body elements.

    Since the field marker may be quite long,
    the second and subsequent lines of the field body do not have to
    line up with the first line, but they must be indented relative to the
    field name marker, and they must line up with each other.
    """
    md.block.ruler.before('paragraph', 'fieldlist', _fieldlist_rule, {'alt': ['paragraph', 'reference', 'blockquote']})