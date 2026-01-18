from fnmatch import fnmatch, fnmatchcase
def match_path_against(pathname, patterns, case_sensitive=True):
    """
    Determines whether the pathname matches any of the given wildcard patterns,
    optionally ignoring the case of the pathname and patterns.

    :param pathname:
        A path name that will be matched against a wildcard pattern.
    :param patterns:
        A list of wildcard patterns to match_path the filename against.
    :param case_sensitive:
        ``True`` if the matching should be case-sensitive; ``False`` otherwise.
    :returns:
        ``True`` if the pattern matches; ``False`` otherwise.

    Doctests::
        >>> match_path_against("/home/username/foobar/blah.py", ["*.py", "*.txt"], False)
        True
        >>> match_path_against("/home/username/foobar/blah.py", ["*.PY", "*.txt"], True)
        False
        >>> match_path_against("/home/username/foobar/blah.py", ["*.PY", "*.txt"], False)
        True
        >>> match_path_against("C:\\windows\\blah\\BLAH.PY", ["*.py", "*.txt"], True)
        False
        >>> match_path_against("C:\\windows\\blah\\BLAH.PY", ["*.py", "*.txt"], False)
        True
    """
    if case_sensitive:
        match_func = fnmatchcase
        pattern_transform_func = lambda w: w
    else:
        match_func = fnmatch
        pathname = pathname.lower()
        pattern_transform_func = _string_lower
    for pattern in set(patterns):
        pattern = pattern_transform_func(pattern)
        if match_func(pathname, pattern):
            return True
    return False