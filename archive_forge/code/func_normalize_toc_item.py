from .util import striptags
def normalize_toc_item(md, token):
    text = token['text']
    tokens = md.inline(text, {})
    html = md.renderer(tokens, {})
    text = striptags(html)
    attrs = token['attrs']
    return (attrs['level'], attrs['id'], text)