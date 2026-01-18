import collections
import re
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import (
def format_rst_table(data, column_names=None):
    """
    Render a table in reStructuredText_ format.

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :returns: The rendered table (a string).

    Here's an example:

    >>> from humanfriendly.tables import format_rst_table
    >>> column_names = ['Version', 'Uploaded on', 'Downloads']
    >>> humanfriendly_releases = [
    ... ['1.23', '2015-05-25', '218'],
    ... ['1.23.1', '2015-05-26', '1354'],
    ... ['1.24', '2015-05-26', '223'],
    ... ['1.25', '2015-05-26', '4319'],
    ... ['1.25.1', '2015-06-02', '197'],
    ... ]
    >>> print(format_rst_table(humanfriendly_releases, column_names))
    =======  ===========  =========
    Version  Uploaded on  Downloads
    =======  ===========  =========
    1.23     2015-05-25   218
    1.23.1   2015-05-26   1354
    1.24     2015-05-26   223
    1.25     2015-05-26   4319
    1.25.1   2015-06-02   197
    =======  ===========  =========

    .. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
    """
    data = [normalize_columns(r) for r in data]
    if column_names:
        data.insert(0, normalize_columns(column_names))
    widths = collections.defaultdict(int)
    for row in data:
        for index, column in enumerate(row):
            widths[index] = max(widths[index], len(column))
    for row in data:
        for index, column in enumerate(row):
            if index < len(row) - 1:
                row[index] = column.ljust(widths[index])
    delimiter = ['=' * w for i, w in sorted(widths.items())]
    if column_names:
        data.insert(1, delimiter)
    data.insert(0, delimiter)
    data.append(delimiter)
    return '\n'.join(('  '.join(r) for r in data))