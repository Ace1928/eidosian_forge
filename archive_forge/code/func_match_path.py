from fnmatch import fnmatch, fnmatchcase
def match_path(pathname, included_patterns=None, excluded_patterns=None, case_sensitive=True):
    """
    Matches a pathname against a set of acceptable and ignored patterns.

    :param pathname:
        A pathname which will be matched against a pattern.
    :param included_patterns:
        Allow filenames matching wildcard patterns specified in this list.
        If no pattern is specified, the function treats the pathname as
        a match_path.
    :param excluded_patterns:
        Ignores filenames matching wildcard patterns specified in this list.
        If no pattern is specified, the function treats the pathname as
        a match_path.
    :param case_sensitive:
        ``True`` if matching should be case-sensitive; ``False`` otherwise.
    :returns:
        ``True`` if the pathname matches; ``False`` otherwise.
    :raises:
        ValueError if included patterns and excluded patterns contain the
        same pattern.

    Doctests::
        >>> match_path("/Users/gorakhargosh/foobar.py")
        True
        >>> match_path("/Users/gorakhargosh/foobar.py", case_sensitive=False)
        True
        >>> match_path("/users/gorakhargosh/foobar.py", ["*.py"], ["*.PY"], True)
        True
        >>> match_path("/users/gorakhargosh/FOOBAR.PY", ["*.py"], ["*.PY"], True)
        False
        >>> match_path("/users/gorakhargosh/foobar/", ["*.py"], ["*.txt"], False)
        False
        >>> match_path("/users/gorakhargosh/FOOBAR.PY", ["*.py"], ["*.PY"], False)
        Traceback (most recent call last):
            ...
        ValueError: conflicting patterns `set(['*.py'])` included and excluded
    """
    included = ['*'] if included_patterns is None else included_patterns
    excluded = [] if excluded_patterns is None else excluded_patterns
    return _match_path(pathname, included, excluded, case_sensitive)