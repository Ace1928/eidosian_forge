def reverse_path(self, crossings=[]):
    """
        Reverse all vertices and arrows of this vertex's component.
        """
    v = self
    while True:
        e = v.in_arrow
        v.reverse()
        if not e:
            break
        e.reverse(crossings)
        v = e.end
        if v == self:
            return
    self.reverse()
    v = self
    while True:
        e = v.out_arrow
        v.reverse()
        if not e:
            break
        e.reverse(crossings)
        v = e.start
        if v == self:
            return