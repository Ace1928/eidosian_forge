def parse_inline_math(inline, m, state):
    text = m.group('math_text')
    state.append_token({'type': 'inline_math', 'raw': text})
    return m.end()