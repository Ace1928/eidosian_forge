from copy import deepcopy
def remove_locus_by_name(self, name):
    """Remove a locus by name."""
    for i, locus in enumerate(self.loci_list):
        if locus == name:
            self.remove_locus_by_position(i)
            return