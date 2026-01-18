from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.helpers import parseLinkLabel
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
def footnote_tail(state: StateCore) -> None:
    """Post-processing step, to move footnote tokens to end of the token stream.

    Also removes un-referenced tokens.
    """
    insideRef = False
    refTokens = {}
    if 'footnotes' not in state.env:
        return
    current: List[Token] = []
    tok_filter = []
    for tok in state.tokens:
        if tok.type == 'footnote_reference_open':
            insideRef = True
            current = []
            currentLabel = tok.meta['label']
            tok_filter.append(False)
            continue
        if tok.type == 'footnote_reference_close':
            insideRef = False
            refTokens[':' + currentLabel] = current
            tok_filter.append(False)
            continue
        if insideRef:
            current.append(tok)
        tok_filter.append(not insideRef)
    state.tokens = [t for t, f in zip(state.tokens, tok_filter) if f]
    if 'list' not in state.env.get('footnotes', {}):
        return
    foot_list = state.env['footnotes']['list']
    token = Token('footnote_block_open', '', 1)
    state.tokens.append(token)
    for i, foot_note in foot_list.items():
        token = Token('footnote_open', '', 1)
        token.meta = {'id': i, 'label': foot_note.get('label', None)}
        state.tokens.append(token)
        if 'tokens' in foot_note:
            tokens = []
            token = Token('paragraph_open', 'p', 1)
            token.block = True
            tokens.append(token)
            token = Token('inline', '', 0)
            token.children = foot_note['tokens']
            token.content = foot_note['content']
            tokens.append(token)
            token = Token('paragraph_close', 'p', -1)
            token.block = True
            tokens.append(token)
        elif 'label' in foot_note:
            tokens = refTokens[':' + foot_note['label']]
        state.tokens.extend(tokens)
        if state.tokens[len(state.tokens) - 1].type == 'paragraph_close':
            lastParagraph: Optional[Token] = state.tokens.pop()
        else:
            lastParagraph = None
        t = foot_note['count'] if 'count' in foot_note and foot_note['count'] > 0 else 1
        j = 0
        while j < t:
            token = Token('footnote_anchor', '', 0)
            token.meta = {'id': i, 'subId': j, 'label': foot_note.get('label', None)}
            state.tokens.append(token)
            j += 1
        if lastParagraph:
            state.tokens.append(lastParagraph)
        token = Token('footnote_close', '', -1)
        state.tokens.append(token)
    token = Token('footnote_block_close', '', -1)
    state.tokens.append(token)