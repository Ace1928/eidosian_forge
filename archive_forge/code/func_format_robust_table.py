import collections
import re
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import (
def format_robust_table(data, column_names):
    """
    Render tabular data with one column per line (allowing columns with line breaks).

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :returns: The rendered table (a string).

    Here's an example:

    >>> from humanfriendly.tables import format_robust_table
    >>> column_names = ['Version', 'Uploaded on', 'Downloads']
    >>> humanfriendly_releases = [
    ... ['1.23', '2015-05-25', '218'],
    ... ['1.23.1', '2015-05-26', '1354'],
    ... ['1.24', '2015-05-26', '223'],
    ... ['1.25', '2015-05-26', '4319'],
    ... ['1.25.1', '2015-06-02', '197'],
    ... ]
    >>> print(format_robust_table(humanfriendly_releases, column_names))
    -----------------------
    Version: 1.23
    Uploaded on: 2015-05-25
    Downloads: 218
    -----------------------
    Version: 1.23.1
    Uploaded on: 2015-05-26
    Downloads: 1354
    -----------------------
    Version: 1.24
    Uploaded on: 2015-05-26
    Downloads: 223
    -----------------------
    Version: 1.25
    Uploaded on: 2015-05-26
    Downloads: 4319
    -----------------------
    Version: 1.25.1
    Uploaded on: 2015-06-02
    Downloads: 197
    -----------------------

    The column names are highlighted in bold font and color so they stand out a
    bit more (see :data:`.HIGHLIGHT_COLOR`).
    """
    blocks = []
    column_names = ['%s:' % n for n in normalize_columns(column_names)]
    if terminal_supports_colors():
        column_names = [highlight_column_name(n) for n in column_names]
    for row in data:
        lines = []
        for column_index, column_text in enumerate(normalize_columns(row)):
            stripped_column = column_text.strip()
            if '\n' not in stripped_column:
                lines.append('%s %s' % (column_names[column_index], stripped_column))
            else:
                lines.append(column_names[column_index])
                lines.extend(column_text.rstrip().splitlines())
        blocks.append(lines)
    num_rows, num_columns = find_terminal_size()
    longest_line = max((max(map(ansi_width, lines)) for lines in blocks))
    delimiter = u'\n%s\n' % ('-' * min(longest_line, num_columns))
    blocks.insert(0, '')
    blocks.append('')
    return delimiter.join((u'\n'.join(b) for b in blocks)).strip()