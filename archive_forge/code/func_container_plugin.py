from __future__ import annotations
from math import floor
from typing import TYPE_CHECKING, Any, Callable, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def container_plugin(md: MarkdownIt, name: str, marker: str=':', validate: None | Callable[[str, str], bool]=None, render: None | Callable[..., str]=None) -> None:
    """Plugin ported from
    `markdown-it-container <https://github.com/markdown-it/markdown-it-container>`__.

    It is a plugin for creating block-level custom containers:

    .. code-block:: md

        :::: name
        ::: name
        *markdown*
        :::
        ::::

    :param name: the name of the container to parse
    :param marker: the marker character to use
    :param validate: func(marker, param) -> bool, default matches against the name
    :param render: render func

    """

    def validateDefault(params: str, *args: Any) -> bool:
        return params.strip().split(' ', 2)[0] == name

    def renderDefault(self: RendererProtocol, tokens: Sequence[Token], idx: int, _options: OptionsDict, env: EnvType) -> str:
        if tokens[idx].nesting == 1:
            tokens[idx].attrJoin('class', name)
        return self.renderToken(tokens, idx, _options, env)
    min_markers = 3
    marker_str = marker
    marker_char = marker_str[0]
    marker_len = len(marker_str)
    validate = validate or validateDefault
    render = render or renderDefault

    def container_func(state: StateBlock, startLine: int, endLine: int, silent: bool) -> bool:
        if is_code_block(state, startLine):
            return False
        auto_closed = False
        start = state.bMarks[startLine] + state.tShift[startLine]
        maximum = state.eMarks[startLine]
        if marker_char != state.src[start]:
            return False
        pos = start + 1
        while pos <= maximum:
            try:
                character = state.src[pos]
            except IndexError:
                break
            if marker_str[(pos - start) % marker_len] != character:
                break
            pos += 1
        marker_count = floor((pos - start) / marker_len)
        if marker_count < min_markers:
            return False
        pos -= (pos - start) % marker_len
        markup = state.src[start:pos]
        params = state.src[pos:maximum]
        assert validate is not None
        if not validate(params, markup):
            return False
        if silent:
            return True
        nextLine = startLine
        while True:
            nextLine += 1
            if nextLine >= endLine:
                break
            start = state.bMarks[nextLine] + state.tShift[nextLine]
            maximum = state.eMarks[nextLine]
            if start < maximum and state.sCount[nextLine] < state.blkIndent:
                break
            if marker_char != state.src[start]:
                continue
            if is_code_block(state, nextLine):
                continue
            pos = start + 1
            while pos <= maximum:
                try:
                    character = state.src[pos]
                except IndexError:
                    break
                if marker_str[(pos - start) % marker_len] != character:
                    break
                pos += 1
            if floor((pos - start) / marker_len) < marker_count:
                continue
            pos -= (pos - start) % marker_len
            pos = state.skipSpaces(pos)
            if pos < maximum:
                continue
            auto_closed = True
            break
        old_parent = state.parentType
        old_line_max = state.lineMax
        state.parentType = 'container'
        state.lineMax = nextLine
        token = state.push(f'container_{name}_open', 'div', 1)
        token.markup = markup
        token.block = True
        token.info = params
        token.map = [startLine, nextLine]
        state.md.block.tokenize(state, startLine + 1, nextLine)
        token = state.push(f'container_{name}_close', 'div', -1)
        token.markup = state.src[start:pos]
        token.block = True
        state.parentType = old_parent
        state.lineMax = old_line_max
        state.line = nextLine + (1 if auto_closed else 0)
        return True
    md.block.ruler.before('fence', 'container_' + name, container_func, {'alt': ['paragraph', 'reference', 'blockquote', 'list']})
    md.add_render_rule(f'container_{name}_open', render)
    md.add_render_rule(f'container_{name}_close', render)