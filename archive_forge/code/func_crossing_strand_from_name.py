import spherogram
def crossing_strand_from_name(link, csname):
    """
    Find crossing strand object from it's name in the format of cslabel above
    """
    for c in link.crossings:
        if c.label == csname[0]:
            return c.crossing_strands()[csname[1]]
    raise ValueError('crossing not found')