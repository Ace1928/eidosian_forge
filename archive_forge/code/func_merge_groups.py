def merge_groups(self):
    """Yield sequence of line groups.  Each one is a tuple:

        'unchanged', lines
             Lines unchanged from base

        'a', lines
             Lines taken from a

        'same', lines
             Lines taken from a (and equal to b)

        'b', lines
             Lines taken from b

        'conflict', base_lines, a_lines, b_lines
             Lines from base were changed to either a or b and conflict.
        """
    for t in self.merge_regions():
        what = t[0]
        if what == 'unchanged':
            yield (what, self.base[t[1]:t[2]])
        elif what == 'a' or what == 'same':
            yield (what, self.a[t[1]:t[2]])
        elif what == 'b':
            yield (what, self.b[t[1]:t[2]])
        elif what == 'conflict':
            yield (what, self.base[t[1]:t[2]], self.a[t[3]:t[4]], self.b[t[5]:t[6]])
        else:
            raise ValueError(what)