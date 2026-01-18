import sys
def docformat(docstring, docdict=None):
    """ Fill a function docstring from variables in dictionary

    Adapt the indent of the inserted docs

    Parameters
    ----------
    docstring : string
        docstring from function, possibly with dict formatting strings
    docdict : dict, optional
        dictionary with keys that match the dict formatting strings
        and values that are docstring fragments to be inserted. The
        indentation of the inserted docstrings is set to match the
        minimum indentation of the ``docstring`` by adding this
        indentation to all lines of the inserted string, except the
        first.

    Returns
    -------
    outstring : string
        string with requested ``docdict`` strings inserted

    Examples
    --------
    >>> docformat(' Test string with %(value)s', {'value':'inserted value'})
    ' Test string with inserted value'
    >>> docstring = 'First line\\n    Second line\\n    %(value)s'
    >>> inserted_string = "indented\\nstring"
    >>> docdict = {'value': inserted_string}
    >>> docformat(docstring, docdict)
    'First line\\n    Second line\\n    indented\\n    string'
    """
    if not docstring:
        return docstring
    if docdict is None:
        docdict = {}
    if not docdict:
        return docstring
    lines = docstring.expandtabs().splitlines()
    if len(lines) < 2:
        icount = 0
    else:
        icount = indentcount_lines(lines[1:])
    indent = ' ' * icount
    indented = {}
    for name, dstr in docdict.items():
        lines = dstr.expandtabs().splitlines()
        try:
            newlines = [lines[0]]
            for line in lines[1:]:
                newlines.append(indent + line)
            indented[name] = '\n'.join(newlines)
        except IndexError:
            indented[name] = dstr
    return docstring % indented