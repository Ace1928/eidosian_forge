import re
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_inline import StateInline
def myst_role_plugin(md: MarkdownIt) -> None:
    """Parse ``{role-name}`content```"""
    md.inline.ruler.before('backticks', 'myst_role', myst_role)
    md.add_render_rule('myst_role', render_myst_role)