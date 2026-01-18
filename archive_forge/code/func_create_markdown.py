from .markdown import Markdown
from .core import BlockState, InlineState, BaseRenderer
from .block_parser import BlockParser
from .inline_parser import InlineParser
from .renderers.html import HTMLRenderer
from .util import escape, escape_url, safe_entity, unikey
from .plugins import import_plugin
def create_markdown(escape: bool=True, hard_wrap: bool=False, renderer='html', plugins=None) -> Markdown:
    """Create a Markdown instance based on the given condition.

    :param escape: Boolean. If using html renderer, escape html.
    :param hard_wrap: Boolean. Break every new line into ``<br>``.
    :param renderer: renderer instance, default is HTMLRenderer.
    :param plugins: List of plugins.

    This method is used when you want to re-use a Markdown instance::

        markdown = create_markdown(
            escape=False,
            hard_wrap=True,
        )
        # re-use markdown function
        markdown('.... your text ...')
    """
    if renderer == 'ast':
        renderer = None
    elif renderer == 'html':
        renderer = HTMLRenderer(escape=escape)
    inline = InlineParser(hard_wrap=hard_wrap)
    if plugins is not None:
        plugins = [import_plugin(n) for n in plugins]
    return Markdown(renderer=renderer, inline=inline, plugins=plugins)