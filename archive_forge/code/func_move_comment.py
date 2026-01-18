from __future__ import unicode_literals
def move_comment(self, target, empty=False):
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
            c = [c[0], c[1], None, None, c[0]]
        target._comment = c
        return self
    if c[0] and tc[0] or (c[1] and tc[1]):
        raise NotImplementedError('overlap in comment %r %r' % (c, tc))
    if c[0]:
        tc[0] = c[0]
    if c[1]:
        tc[1] = c[1]
    return self