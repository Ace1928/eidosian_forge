from __future__ import annotations
from ruamel.yaml.compat import nprintf  # NOQA
from .error import StreamMark  # NOQA
def move_new_comment(self, target: Any, empty: bool=False) -> Any:
    """move a comment from this token to target (normally next token)
        used to combine e.g. comments before a BlockEntryToken to the
        ScalarToken that follows it
        empty is a special for empty values -> comment after key
        """
    c = self.comment
    if c is None:
        return
    if isinstance(target, (StreamEndToken, DocumentStartToken)):
        return
    delattr(self, '_comment')
    tc = target.comment
    if not tc:
        if empty:
            c = [c[0], c[1], c[2]]
        target._comment = c
        return self
    for idx in range(3):
        if c[idx] is not None and tc[idx] is not None:
            raise NotImplementedError(f'overlap in comment {c!r} {tc!r}')
    for idx in range(3):
        if c[idx]:
            tc[idx] = c[idx]
    return self