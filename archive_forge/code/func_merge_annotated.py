def merge_annotated(self):
    """Return merge with conflicts, showing origin of lines.

        Most useful for debugging merge.
        """
    if self._uses_bytes():
        UNCHANGED = b'u'
        SEP = b' | '
        CONFLICT_START = b'<<<<\n'
        CONFLICT_MID = b'----\n'
        CONFLICT_END = b'>>>>\n'
        WIN_A = b'a'
        WIN_B = b'b'
    else:
        UNCHANGED = 'u'
        SEP = ' | '
        CONFLICT_START = '<<<<\n'
        CONFLICT_MID = '----\n'
        CONFLICT_END = '>>>>\n'
        WIN_A = 'a'
        WIN_B = 'b'
    for t in self.merge_regions():
        what = t[0]
        if what == 'unchanged':
            for i in range(t[1], t[2]):
                yield (UNCHANGED + SEP + self.base[i])
        elif what == 'a' or what == 'same':
            for i in range(t[1], t[2]):
                yield (WIN_A.lower() + SEP + self.a[i])
        elif what == 'b':
            for i in range(t[1], t[2]):
                yield (WIN_B.lower() + SEP + self.b[i])
        elif what == 'conflict':
            yield CONFLICT_START
            for i in range(t[3], t[4]):
                yield (WIN_A.upper() + SEP + self.a[i])
            yield CONFLICT_MID
            for i in range(t[5], t[6]):
                yield (WIN_B.upper() + SEP + self.b[i])
            yield CONFLICT_END
        else:
            raise ValueError(what)