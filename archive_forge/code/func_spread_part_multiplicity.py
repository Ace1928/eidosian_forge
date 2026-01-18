important invariant is that the parts on the stack are themselves in
def spread_part_multiplicity(self):
    """Returns True if a new part has been created, and
        adjusts pstack, f and lpart as needed.

        Notes
        =====

        Spreads unallocated multiplicity from the current top part
        into a new part created above the current on the stack.  This
        new part is constrained to be less than or equal to the old in
        terms of the part ordering.

        This call does nothing (and returns False) if the current top
        part has no unallocated multiplicity.

        """
    j = self.f[self.lpart]
    k = self.f[self.lpart + 1]
    base = k
    changed = False
    for j in range(self.f[self.lpart], self.f[self.lpart + 1]):
        self.pstack[k].u = self.pstack[j].u - self.pstack[j].v
        if self.pstack[k].u == 0:
            changed = True
        else:
            self.pstack[k].c = self.pstack[j].c
            if changed:
                self.pstack[k].v = self.pstack[k].u
            elif self.pstack[k].u < self.pstack[j].v:
                self.pstack[k].v = self.pstack[k].u
                changed = True
            else:
                self.pstack[k].v = self.pstack[j].v
            k = k + 1
    if k > base:
        self.lpart = self.lpart + 1
        self.f[self.lpart + 1] = k
        return True
    return False