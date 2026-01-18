from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.helpers import parseLinkLabel
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
def render_footnote_open(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    ident: str = self.rules['footnote_anchor_name'](tokens, idx, options, env)
    if tokens[idx].meta.get('subId', -1) > 0:
        ident += ':' + tokens[idx].meta['subId']
    return '<li id="fn' + ident + '" class="footnote-item">'