from __future__ import annotations
from collections.abc import Sequence
import inspect
from typing import Any, ClassVar, Protocol
from .common.utils import escapeHtml, unescapeAll
from .token import Token
from .utils import EnvType, OptionsDict
def renderInlineAsText(self, tokens: Sequence[Token] | None, options: OptionsDict, env: EnvType) -> str:
    """Special kludge for image `alt` attributes to conform CommonMark spec.

        Don't try to use it! Spec requires to show `alt` content with stripped markup,
        instead of simple escaping.

        :param tokens: list on block tokens to render
        :param options: params of parser instance
        :param env: additional data from parsed input
        """
    result = ''
    for token in tokens or []:
        if token.type == 'text':
            result += token.content
        elif token.type == 'image':
            if token.children:
                result += self.renderInlineAsText(token.children, options, env)
        elif token.type == 'softbreak':
            result += '\n'
    return result