import logging
from ..common.utils import isStrSpace
from .state_block import StateBlock
def markTightParagraphs(state: StateBlock, idx: int) -> None:
    level = state.level + 2
    i = idx + 2
    length = len(state.tokens) - 2
    while i < length:
        if state.tokens[i].level == level and state.tokens[i].type == 'paragraph_open':
            state.tokens[i + 2].hidden = True
            state.tokens[i].hidden = True
            i += 2
        i += 1