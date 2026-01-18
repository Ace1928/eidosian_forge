import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def conllstr2tree(s, chunk_types=('NP', 'PP', 'VP'), root_label='S'):
    """
    Return a chunk structure for a single sentence
    encoded in the given CONLL 2000 style string.
    This function converts a CoNLL IOB string into a tree.
    It uses the specified chunk types
    (defaults to NP, PP and VP), and creates a tree rooted at a node
    labeled S (by default).

    :param s: The CoNLL string to be converted.
    :type s: str
    :param chunk_types: The chunk types to be converted.
    :type chunk_types: tuple
    :param root_label: The node label to use for the root.
    :type root_label: str
    :rtype: Tree
    """
    stack = [Tree(root_label, [])]
    for lineno, line in enumerate(s.split('\n')):
        if not line.strip():
            continue
        match = _LINE_RE.match(line)
        if match is None:
            raise ValueError(f'Error on line {lineno:d}')
        word, tag, state, chunk_type = match.groups()
        if chunk_types is not None and chunk_type not in chunk_types:
            state = 'O'
        mismatch_I = state == 'I' and chunk_type != stack[-1].label()
        if state in 'BO' or mismatch_I:
            if len(stack) == 2:
                stack.pop()
        if state == 'B' or mismatch_I:
            chunk = Tree(chunk_type, [])
            stack[-1].append(chunk)
            stack.append(chunk)
        stack[-1].append((word, tag))
    return stack[0]