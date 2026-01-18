def suffix_replace(original, old, new):
    """
    Replaces the old suffix of the original string by a new suffix
    """
    return original[:-len(old)] + new