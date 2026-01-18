from ..util import escape_url
def parse_url_link(inline, m, state):
    text = m.group(0)
    pos = m.end()
    if state.in_link:
        inline.process_text(text, state)
        return pos
    state.append_token({'type': 'link', 'children': [{'type': 'text', 'raw': text}], 'attrs': {'url': escape_url(text)}})
    return pos