import sys
def replace_notes_in_docstring(cls, notes):
    """
    This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It replaces the 'Notes' section of that docstring with
    the given `notes`.
    """

    def _doc(func):
        cls_docstring = getattr(cls, func.__name__).__doc__
        notes_header = '        Notes\n        -----\n'
        if cls_docstring is None:
            return func
        start_of_notes = cls_docstring.find(notes_header)
        end_of_notes = cls_docstring.find('        References\n')
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find('        Examples\n')
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        func.__doc__ = cls_docstring[:start_of_notes + len(notes_header)] + notes + cls_docstring[end_of_notes:]
        return func
    return _doc