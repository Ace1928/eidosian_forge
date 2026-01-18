import re
def spoiler(md):
    """A mistune plugin to support block and inline spoiler. The
    syntax is inspired by stackexchange:

    .. code-block:: text

        Block level spoiler looks like block quote, but with `>!`:

        >! this is spoiler
        >!
        >! the content will be hidden

        Inline spoiler is surrounded by `>!` and `!<`, such as >! hide me !<.

    :param md: Markdown instance
    """
    md.block.register('block_quote', None, parse_block_spoiler)
    md.inline.register('inline_spoiler', INLINE_SPOILER_PATTERN, parse_inline_spoiler)
    if md.renderer and md.renderer.NAME == 'html':
        md.renderer.register('block_spoiler', render_block_spoiler)
        md.renderer.register('inline_spoiler', render_inline_spoiler)