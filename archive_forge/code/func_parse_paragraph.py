import re
import string
def parse_paragraph(block, m, state):
    text = m.group(0)
    state.add_paragraph(text)
    return m.end()