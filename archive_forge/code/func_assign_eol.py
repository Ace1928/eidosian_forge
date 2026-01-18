from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
def assign_eol(self, tokens: Any) -> Any:
    try:
        comment_line = self.unused[0]
    except IndexError:
        return
    if not isinstance(self.comments[comment_line], EOLComment):
        return
    idx = 1
    while tokens[-idx].start_mark.line > comment_line or isinstance(tokens[-idx], ValueToken):
        idx += 1
    if _debug != 0:
        xprintf('idx1', idx)
    if len(tokens) > idx and isinstance(tokens[-idx], ScalarToken) and isinstance(tokens[-(idx + 1)], ScalarToken):
        return
    try:
        if isinstance(tokens[-idx], ScalarToken) and isinstance(tokens[-(idx + 1)], KeyToken):
            try:
                eol_idx = self.unused.pop(0)
                self.comments[eol_idx].set_used()
                if _debug != 0:
                    xprintf('>>>>>a', idx, eol_idx, KEYCMNT)
                tokens[-idx].add_comment_eol(eol_idx, KEYCMNT)
            except IndexError:
                raise NotImplementedError
            return
    except IndexError:
        if _debug != 0:
            xprintf('IndexError1')
        pass
    try:
        if isinstance(tokens[-idx], ScalarToken) and isinstance(tokens[-(idx + 1)], (ValueToken, BlockEntryToken)):
            try:
                eol_idx = self.unused.pop(0)
                self.comments[eol_idx].set_used()
                tokens[-idx].add_comment_eol(eol_idx, VALUECMNT)
            except IndexError:
                raise NotImplementedError
            return
    except IndexError:
        if _debug != 0:
            xprintf('IndexError2')
        pass
    for t in tokens:
        xprintf('tt-', t)
    if _debug != 0:
        xprintf('not implemented EOL', type(tokens[-idx]))
    import sys
    sys.exit(0)