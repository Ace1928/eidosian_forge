from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.scanner import Scanner, RoundTripScanner, ScannerError  # NOQA
from ruamel.yaml.scanner import BlankLineComment
from ruamel.yaml.comments import C_PRE, C_POST, C_SPLIT_ON_FIRST_BLANK
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.tag import Tag
def distribute_comment(self, comment: Any, line: Any) -> Any:
    if comment is None:
        return None
    if not comment[0]:
        return None
    assert comment[0][0] == line + 1
    typ = self.loader.comment_handling & 3
    if typ == C_POST:
        return None
    if typ == C_PRE:
        c = [None, None, comment[0]]
        comment[0] = None
        return c
    for _idx, cmntidx in enumerate(comment[0]):
        if isinstance(self.scanner.comments[cmntidx], BlankLineComment):
            break
    else:
        return None
    if _idx == 0:
        return None
    if typ == C_SPLIT_ON_FIRST_BLANK:
        c = [None, None, comment[0][:_idx]]
        comment[0] = comment[0][_idx:]
        return c
    raise NotImplementedError