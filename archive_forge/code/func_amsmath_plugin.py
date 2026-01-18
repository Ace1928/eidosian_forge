from __future__ import annotations
import re
from typing import TYPE_CHECKING, Callable, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def amsmath_plugin(md: MarkdownIt, *, renderer: Optional[Callable[[str], str]]=None) -> None:
    """Parses TeX math equations, without any surrounding delimiters,
    only for top-level `amsmath <https://ctan.org/pkg/amsmath>`__ environments:

    .. code-block:: latex

        \\begin{gather*}
        a_1=b_1+c_1\\\\
        a_2=b_2+c_2-d_2+e_2
        \\end{gather*}

    :param renderer: Function to render content, by default escapes HTML

    """
    md.block.ruler.before('blockquote', 'amsmath', amsmath_block, {'alt': ['paragraph', 'reference', 'blockquote', 'list', 'footnote_def']})
    _renderer = (lambda content: escapeHtml(content)) if renderer is None else renderer

    def render_amsmath_block(self: RendererProtocol, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType) -> str:
        content = _renderer(str(tokens[idx].content))
        return f'<div class="math amsmath">\n{content}\n</div>\n'
    md.add_render_rule('amsmath', render_amsmath_block)