import ast
from nbconvert.preprocessors import Preprocessor
def filter_magic(source, magic, strip=True):
    """
    Given the source of a cell, filter out the given magic and collect
    the lines using the magic into a list.

    If strip is True, the IPython syntax part of the magic (e.g. %magic
    or %%magic) is stripped from the returned lines.
    """
    filtered, magic_lines = ([], [])
    for line in source.splitlines():
        if line.strip().startswith(magic):
            magic_lines.append(line)
        else:
            filtered.append(line)
    if strip:
        magic_lines = [el.replace(magic, '') for el in magic_lines]
    return ('\n'.join(filtered), magic_lines)