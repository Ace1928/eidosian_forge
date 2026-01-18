from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def skipMarker(state: StateBlock, line: int) -> int:
    """Search `[:~][
 ]`, returns next pos after marker on success or -1 on fail."""
    start = state.bMarks[line] + state.tShift[line]
    maximum = state.eMarks[line]
    if start >= maximum:
        return -1
    marker = state.src[start]
    start += 1
    if marker != '~' and marker != ':':
        return -1
    pos = state.skipSpaces(start)
    if start == pos:
        return -1
    if pos >= maximum:
        return -1
    return start