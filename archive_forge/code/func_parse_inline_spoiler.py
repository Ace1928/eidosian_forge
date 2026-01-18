import re
def parse_inline_spoiler(inline, m, state):
    text = m.group('spoiler_text')
    new_state = state.copy()
    new_state.src = text
    children = inline.render(new_state)
    state.append_token({'type': 'inline_spoiler', 'children': children})
    return m.end()