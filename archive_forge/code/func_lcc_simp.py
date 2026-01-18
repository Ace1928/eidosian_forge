import math
def lcc_simp(seq):
    """Calculate Local Composition Complexity (LCC) for a sequence.

    seq - an unambiguous DNA sequence (a string or Seq object)

    Returns the Local Composition Complexity (LCC) value for the entire
    sequence (as a float).

    Reference:
    Andrzej K Konopka (2005) Sequence Complexity and Composition
    https://doi.org/10.1038/npg.els.0005260
    """
    wsize = len(seq)
    seq = seq.upper()
    l4 = math.log(4)
    if 'A' not in seq:
        term_a = 0
    else:
        term_a = seq.count('A') / wsize * math.log(seq.count('A') / wsize) / l4
    if 'C' not in seq:
        term_c = 0
    else:
        term_c = seq.count('C') / wsize * math.log(seq.count('C') / wsize) / l4
    if 'T' not in seq:
        term_t = 0
    else:
        term_t = seq.count('T') / wsize * math.log(seq.count('T') / wsize) / l4
    if 'G' not in seq:
        term_g = 0
    else:
        term_g = seq.count('G') / wsize * math.log(seq.count('G') / wsize) / l4
    return -(term_a + term_c + term_t + term_g)