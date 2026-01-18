import re
def hit_coverage(self):
    """Return the length of the hit covered in the alignment."""
    s = self.hit_aln.replace('=', '')
    return len(s)