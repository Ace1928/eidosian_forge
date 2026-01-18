def struct_to_lines(self, struct_iter):
    """Convert merge result tuples to lines"""
    for lines in struct_iter:
        if len(lines) == 1:
            yield from lines[0]
        else:
            yield self.a_marker
            yield from lines[0]
            yield self.split_marker
            yield from lines[1]
            yield self.b_marker