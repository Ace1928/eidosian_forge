import re
from ..core import BlockState
from ..util import unikey
from ..helpers import LINK_LABEL
def parse_footnote_item(block, key: str, index: int, state: BlockState):
    ref = state.env.get('ref_footnotes')
    text = ref[key]
    lines = text.splitlines()
    second_line = None
    for second_line in lines[1:]:
        if second_line:
            break
    if second_line:
        spaces = len(second_line) - len(second_line.lstrip())
        pattern = re.compile('^ {' + str(spaces) + ',}', flags=re.M)
        text = pattern.sub('', text).strip()
        items = _PARAGRAPH_SPLIT.split(text)
        children = [{'type': 'paragraph', 'text': s} for s in items]
    else:
        text = text.strip()
        children = [{'type': 'paragraph', 'text': text}]
    return {'type': 'footnote_item', 'children': children, 'attrs': {'key': key, 'index': index}}