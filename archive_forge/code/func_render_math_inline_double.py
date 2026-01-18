from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, isWhiteSpace
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
def render_math_inline_double(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
    content = _renderer(str(tokens[idx].content).strip(), {'display_mode': True})
    return f'<div class="math inline">{content}</div>'