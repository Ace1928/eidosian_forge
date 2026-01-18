import sys
from fnmatch import fnmatchcase
from pathlib import Path
def find_package_data(where='.', package='', exclude=standard_exclude, exclude_directories=standard_exclude_directories, only_in_packages=True, show_ignored=False):
    """
    Return a dictionary suitable for use in ``package_data``
    in a distutils ``setup.py`` file.

    The dictionary looks like::

        {'package': [files]}

    Where ``files`` is a list of all the files in that package that
    don't match anything in ``exclude``.

    If ``only_in_packages`` is true, then top-level directories that
    are not packages won't be included (but directories under packages
    will).

    Directories matching any pattern in ``exclude_directories`` will
    be ignored; by default directories with leading ``.``, ``CVS``,
    and ``_darcs`` will be ignored.

    If ``show_ignored`` is true, then all the files that aren't
    included in package data are shown on stderr (for debugging
    purposes).

    Note patterns use wildcards, or can be exact paths (including
    leading ``./``), and all searching is case-insensitive.
    """
    out = {}
    stack = [(Path(where), '', package, only_in_packages)]
    while stack:
        where, prefix, package, only_in_packages = stack.pop(0)
        for name in where.iterdir():
            fn = where.joinpath(name)
            if fn.is_dir():
                bad_name = False
                for pattern in exclude_directories:
                    if fnmatchcase(name.as_posix(), pattern) or fn.as_posix().lower() == pattern.lower():
                        bad_name = True
                        if show_ignored:
                            print('Directory %s ignored by pattern %s' % (fn.as_posix(), pattern), file=sys.stderr)
                        break
                if bad_name:
                    continue
                if fn.joinpath('__init__.py').is_file() and (not prefix):
                    if not package:
                        new_package = name.as_posix()
                    else:
                        new_package = package + '.' + name.as_posix()
                    stack.append((fn, '', new_package, False))
                else:
                    stack.append((fn, prefix + name.as_posix() + '/', package, only_in_packages))
            elif package or not only_in_packages:
                bad_name = False
                for pattern in exclude:
                    if fnmatchcase(name.as_posix(), pattern) or fn.as_posix().lower() == pattern.lower():
                        bad_name = True
                        if show_ignored:
                            print('File %s ignored by pattern %s' % (fn.as_posix(), pattern), file=sys.stderr)
                        break
                if bad_name:
                    continue
                out.setdefault(package, []).append(prefix + name.as_posix())
    return out