from contextlib import contextmanager
import sys
import warnings
import re
import functools
import os
@contextmanager
def expected_warnings(matching):
    """Context for use in testing to catch known warnings matching regexes

    Parameters
    ----------
    matching : None or a list of strings or compiled regexes
        Regexes for the desired warning to catch
        If matching is None, this behaves as a no-op.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> image = rng.integers(0, 2**16, size=(100, 100), dtype=np.uint16)
    >>> # rank filters are slow when bit-depth exceeds 10 bits
    >>> from skimage import filters
    >>> with expected_warnings(['Bad rank filter performance']):
    ...     median_filtered = filters.rank.median(image)

    Notes
    -----
    Uses `all_warnings` to ensure all warnings are raised.
    Upon exiting, it checks the recorded warnings for the desired matching
    pattern(s).
    Raises a ValueError if any match was not found or an unexpected
    warning was raised.
    Allows for three types of behaviors: `and`, `or`, and `optional` matches.
    This is done to accommodate different build environments or loop conditions
    that may produce different warnings.  The behaviors can be combined.
    If you pass multiple patterns, you get an orderless `and`, where all of the
    warnings must be raised.
    If you use the `|` operator in a pattern, you can catch one of several
    warnings.
    Finally, you can use `|\\A\\Z` in a pattern to signify it as optional.

    """
    if isinstance(matching, str):
        raise ValueError('``matching`` should be a list of strings and not a string itself.')
    if matching is None:
        yield None
        return
    strict_warnings = os.environ.get('SKIMAGE_TEST_STRICT_WARNINGS', '1')
    if strict_warnings.lower() == 'true':
        strict_warnings = True
    elif strict_warnings.lower() == 'false':
        strict_warnings = False
    else:
        strict_warnings = bool(int(strict_warnings))
    with all_warnings() as w:
        yield w
        while None in matching:
            matching.remove(None)
        remaining = [m for m in matching if '\\A\\Z' not in m.split('|')]
        for warn in w:
            found = False
            for match in matching:
                if re.search(match, str(warn.message)) is not None:
                    found = True
                    if match in remaining:
                        remaining.remove(match)
            if strict_warnings and (not found):
                raise ValueError(f'Unexpected warning: {str(warn.message)}')
        if strict_warnings and len(remaining) > 0:
            newline = '\n'
            msg = f'No warning raised matching:{newline}{newline.join(remaining)}'
            raise ValueError(msg)