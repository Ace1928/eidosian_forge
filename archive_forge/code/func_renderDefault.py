from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def renderDefault(self: RendererProtocol, tokens: Sequence[Token], idx: int, _options: OptionsDict, env: EnvType) -> str:
    return self.renderToken(tokens, idx, _options, env)