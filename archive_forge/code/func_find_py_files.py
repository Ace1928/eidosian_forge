import os
import re
import sysconfig
def find_py_files(sources, recursive, exclude=None):
    """Find Python source files.

    Parameters
        - sources: iterable with paths as strings.
        - recursive: drill down directories if True.
        - exclude: string based on which directories and files are excluded.

    Return: yields paths to found files.
    """

    def not_hidden(name):
        """Return True if file 'name' isn't .hidden."""
        return not name.startswith('.')

    def is_excluded(name, exclude):
        """Return True if file 'name' is excluded."""
        return any((re.search(re.escape(str(e)), name, re.IGNORECASE) for e in exclude)) if exclude else False
    for name in sorted(sources):
        if recursive and os.path.isdir(name):
            for root, dirs, children in os.walk(unicode(name)):
                dirs[:] = [d for d in dirs if not_hidden(d) and (not is_excluded(d, _PYTHON_LIBS))]
                dirs[:] = sorted([d for d in dirs if not is_excluded(d, exclude)])
                files = sorted([f for f in children if not_hidden(f) and (not is_excluded(f, exclude))])
                for filename in files:
                    if filename.endswith('.py') and (not is_excluded(root, exclude)):
                        yield os.path.join(root, filename)
        else:
            yield name