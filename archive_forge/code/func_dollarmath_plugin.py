from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, isWhiteSpace
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline
from mdit_py_plugins.utils import is_code_block
def dollarmath_plugin(md: MarkdownIt, *, allow_labels: bool=True, allow_space: bool=True, allow_digits: bool=True, allow_blank_lines: bool=True, double_inline: bool=False, label_normalizer: Optional[Callable[[str], str]]=None, renderer: Optional[Callable[[str, Dict[str, Any]], str]]=None, label_renderer: Optional[Callable[[str], str]]=None) -> None:
    """Plugin for parsing dollar enclosed math,
    e.g. inline: ``$a=1$``, block: ``$$b=2$$``

    This is an improved version of ``texmath``; it is more performant,
    and handles ``\\`` escaping properly and allows for more configuration.

    :param allow_labels: Capture math blocks with label suffix, e.g. ``$$a=1$$ (eq1)``
    :param allow_space: Parse inline math when there is space
        after/before the opening/closing ``$``, e.g. ``$ a $``
    :param allow_digits: Parse inline math when there is a digit
        before/after the opening/closing ``$``, e.g. ``1$`` or ``$2``.
        This is useful when also using currency.
    :param allow_blank_lines: Allow blank lines inside ``$$``. Note that blank lines are
        not allowed in LaTeX, executablebooks/markdown-it-dollarmath, or the Github or
        StackExchange markdown dialects. Hoever, they have special semantics if used
        within Sphinx `..math` admonitions, so are allowed for backwards-compatibility.
    :param double_inline: Search for double-dollar math within inline contexts
    :param label_normalizer: Function to normalize the label,
        by default replaces whitespace with `-`
    :param renderer: Function to render content: `(str, {"display_mode": bool}) -> str`,
        by default escapes HTML
    :param label_renderer: Function to render labels, by default creates anchor

    """
    if label_normalizer is None:
        label_normalizer = lambda label: re.sub('\\s+', '-', label)
    md.inline.ruler.before('escape', 'math_inline', math_inline_dollar(allow_space, allow_digits, double_inline))
    md.block.ruler.before('fence', 'math_block', math_block_dollar(allow_labels, label_normalizer, allow_blank_lines))
    _renderer = (lambda content, _: escapeHtml(content)) if renderer is None else renderer
    _label_renderer: Callable[[str], str]
    if label_renderer is None:
        _label_renderer = lambda label: f'<a href="#{label}" class="mathlabel" title="Permalink to this equation">¶</a>'
    else:
        _label_renderer = label_renderer

    def render_math_inline(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
        content = _renderer(str(tokens[idx].content).strip(), {'display_mode': False})
        return f'<span class="math inline">{content}</span>'

    def render_math_inline_double(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
        content = _renderer(str(tokens[idx].content).strip(), {'display_mode': True})
        return f'<div class="math inline">{content}</div>'

    def render_math_block(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
        content = _renderer(str(tokens[idx].content).strip(), {'display_mode': True})
        return f'<div class="math block">\n{content}\n</div>\n'

    def render_math_block_label(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
        content = _renderer(str(tokens[idx].content).strip(), {'display_mode': True})
        _id = tokens[idx].info
        label = _label_renderer(tokens[idx].info)
        return f'<div id="{_id}" class="math block">\n{label}\n{content}\n</div>\n'
    md.add_render_rule('math_inline', render_math_inline)
    md.add_render_rule('math_inline_double', render_math_inline_double)
    md.add_render_rule('math_block', render_math_block)
    md.add_render_rule('math_block_label', render_math_block_label)