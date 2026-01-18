def condition_with_prefix(prefix, condition):
    """Returns the given string prefixed by the given prefix.

    If the prefix is non-empty, a colon is used to separate them.
    """
    if prefix == '' or prefix is None:
        return condition
    return prefix + ':' + condition