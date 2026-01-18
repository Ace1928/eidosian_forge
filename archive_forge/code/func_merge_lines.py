def merge_lines(self, name_a=None, name_b=None, name_base=None, start_marker=None, mid_marker=None, end_marker=None, base_marker=None, reprocess=False):
    """Return merge in cvs-like form."""
    if base_marker and reprocess:
        raise CantReprocessAndShowBase()
    if self._uses_bytes():
        if len(self.a) > 0:
            if self.a[0].endswith(b'\r\n'):
                newline = b'\r\n'
            elif self.a[0].endswith(b'\r'):
                newline = b'\r'
            else:
                newline = b'\n'
        else:
            newline = b'\n'
        if start_marker is None:
            start_marker = b'<<<<<<<'
        if mid_marker is None:
            mid_marker = b'======='
        if end_marker is None:
            end_marker = b'>>>>>>>'
        space = b' '
    else:
        if start_marker is None:
            start_marker = '<<<<<<<'
        if mid_marker is None:
            mid_marker = '======='
        if end_marker is None:
            end_marker = '>>>>>>>'
        if len(self.a) > 0:
            if self.a[0].endswith('\r\n'):
                newline = '\r\n'
            elif self.a[0].endswith('\r'):
                newline = '\r'
            else:
                newline = '\n'
        else:
            newline = '\n'
        space = ' '
    if name_a:
        start_marker = start_marker + space + name_a
    if name_b:
        end_marker = end_marker + space + name_b
    if name_base and base_marker:
        base_marker = base_marker + space + name_base
    merge_regions = self.merge_regions()
    if reprocess is True:
        merge_regions = self.reprocess_merge_regions(merge_regions)
    for t in merge_regions:
        what = t[0]
        if what == 'unchanged':
            for i in range(t[1], t[2]):
                yield self.base[i]
        elif what == 'a' or what == 'same':
            for i in range(t[1], t[2]):
                yield self.a[i]
        elif what == 'b':
            for i in range(t[1], t[2]):
                yield self.b[i]
        elif what == 'conflict':
            yield (start_marker + newline)
            for i in range(t[3], t[4]):
                yield self.a[i]
            if base_marker is not None:
                yield (base_marker + newline)
                for i in range(t[1], t[2]):
                    yield self.base[i]
            yield (mid_marker + newline)
            for i in range(t[5], t[6]):
                yield self.b[i]
            yield (end_marker + newline)
        else:
            raise ValueError(what)