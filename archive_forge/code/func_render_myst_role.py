import re
from typing import TYPE_CHECKING, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_inline import StateInline
def render_myst_role(self: 'RendererProtocol', tokens: Sequence['Token'], idx: int, options: 'OptionsDict', env: 'EnvType') -> str:
    token = tokens[idx]
    name = token.meta.get('name', 'unknown')
    return f'<code class="myst role">{{{name}}}[{escapeHtml(token.content)}]</code>'