def english_lower(s):
    """ Apply English case rules to convert ASCII strings to all lower case.

    This is an internal utility function to replace calls to str.lower() such
    that we can avoid changing behavior with changing locales. In particular,
    Turkish has distinct dotted and dotless variants of the Latin letter "I" in
    both lowercase and uppercase. Thus, "I".lower() != "i" in a "tr" locale.

    Parameters
    ----------
    s : str

    Returns
    -------
    lowered : str

    Examples
    --------
    >>> from numpy.core.numerictypes import english_lower
    >>> english_lower('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
    'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz0123456789_'
    >>> english_lower('')
    ''
    """
    lowered = s.translate(LOWER_TABLE)
    return lowered