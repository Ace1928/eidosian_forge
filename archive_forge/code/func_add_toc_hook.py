from .util import striptags
def add_toc_hook(md, min_level=1, max_level=3, heading_id=None):
    """Add a hook to save toc items into ``state.env``. This is
    usually helpful for doc generator::

        import mistune
        from mistune.toc import add_toc_hook, render_toc_ul

        md = mistune.create_markdown(...)
        add_toc_hook(md)

        html, state = md.parse(text)
        toc_items = state.env['toc_items']
        toc_html = render_toc_ul(toc_items)

    :param md: Markdown instance
    :param min_level: min heading level
    :param max_level: max heading level
    :param heading_id: a function to generate heading_id
    """
    if heading_id is None:

        def heading_id(token, index):
            return 'toc_' + str(index + 1)

    def toc_hook(md, state):
        headings = []
        for tok in state.tokens:
            if tok['type'] == 'heading':
                level = tok['attrs']['level']
                if min_level <= level <= max_level:
                    headings.append(tok)
        toc_items = []
        for i, tok in enumerate(headings):
            tok['attrs']['id'] = heading_id(tok, i)
            toc_items.append(normalize_toc_item(md, tok))
        state.env['toc_items'] = toc_items
    md.before_render_hooks.append(toc_hook)