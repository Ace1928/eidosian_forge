from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.helpers import parseLinkLabel
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
def footnote_plugin(md: MarkdownIt) -> None:
    """Plugin ported from
    `markdown-it-footnote <https://github.com/markdown-it/markdown-it-footnote>`__.

    It is based on the
    `pandoc definition <http://johnmacfarlane.net/pandoc/README.html#footnotes>`__:

    .. code-block:: md

        Normal footnote:

        Here is a footnote reference,[^1] and another.[^longnote]

        [^1]: Here is the footnote.

        [^longnote]: Here's one with multiple blocks.

            Subsequent paragraphs are indented to show that they
        belong to the previous footnote.

    """
    md.block.ruler.before('reference', 'footnote_def', footnote_def, {'alt': ['paragraph', 'reference']})
    md.inline.ruler.after('image', 'footnote_inline', footnote_inline)
    md.inline.ruler.after('footnote_inline', 'footnote_ref', footnote_ref)
    md.core.ruler.after('inline', 'footnote_tail', footnote_tail)
    md.add_render_rule('footnote_ref', render_footnote_ref)
    md.add_render_rule('footnote_block_open', render_footnote_block_open)
    md.add_render_rule('footnote_block_close', render_footnote_block_close)
    md.add_render_rule('footnote_open', render_footnote_open)
    md.add_render_rule('footnote_close', render_footnote_close)
    md.add_render_rule('footnote_anchor', render_footnote_anchor)
    md.add_render_rule('footnote_caption', render_footnote_caption)
    md.add_render_rule('footnote_anchor_name', render_footnote_anchor_name)