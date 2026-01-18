import re
from ._base import DirectiveParser, BaseDirective
def parse_fenced_code(self, block, m, state):
    info = m.group('fenced_3')
    if not info or not _type_re.match(info):
        return block.parse_fenced_code(m, state)
    if state.depth() >= block.max_nested_level:
        return block.parse_fenced_code(m, state)
    marker = m.group('fenced_2')
    return self._process_directive(block, marker, m.start(), state)