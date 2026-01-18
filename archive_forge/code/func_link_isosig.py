import spherogram
def link_isosig(link, root=None, over_or_under=False):
    """
    A list of data which encodes a planar isotopy class of links instead of
    tangles.  This is just the minimal isosig gotten by cutting all possible
    strands to get a tangle with 1 strand.
    """
    return min((cut_strand(link, cs).isosig(root, over_or_under) for cs in link.crossing_strands()))