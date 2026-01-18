from __future__ import print_function, unicode_literals
import sys
import typing
from fs.path import abspath, join, normpath
def format_directory(path, levels):
    """Recursive directory function."""
    try:
        directory = sorted(fs.filterdir(path, exclude_dirs=exclude, files=filter), key=sort_key_dirs_first if dirs_first else sort_key)
    except Exception as error:
        prefix = ''.join((indent if last else line_indent for last in levels)) + char_corner + char_line
        write('{} {}'.format(format_prefix(prefix), format_error('error ({})'.format(error))))
        return
    _last = len(directory) - 1
    for i, info in enumerate(directory):
        is_last_entry = i == _last
        counts['dirs' if info.is_dir else 'files'] += 1
        prefix = ''.join((indent if last else line_indent for last in levels))
        prefix += char_corner if is_last_entry else char_newnode
        if info.is_dir:
            write('{} {}'.format(format_prefix(prefix + char_line), format_dirname(info.name)))
            if max_levels is None or len(levels) < max_levels:
                format_directory(join(path, info.name), levels + [is_last_entry])
        else:
            write('{} {}'.format(format_prefix(prefix + char_line), format_filename(info.name)))