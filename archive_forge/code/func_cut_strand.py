import spherogram
def cut_strand(link, cs):
    """
    Cut a link along a strand to get a tangle with 1 strand
    """
    link_copy = link.copy()
    cs_copy = crossing_strand_from_name(link_copy, cslabel(cs))
    op = cs_copy.opposite()
    cs_copy.crossing.adjacent[cs_copy.strand_index] = None
    op.crossing.adjacent[op.strand_index] = None
    return spherogram.Tangle(1, link_copy.crossings, [(cs_copy.crossing, cs_copy.strand_index), (op.crossing, op.strand_index)])