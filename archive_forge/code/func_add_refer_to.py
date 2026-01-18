from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
def add_refer_to(method):
    """
    Build decorator which appends link to the high-level equivalent method to the function's docstring.

    Parameters
    ----------
    method : str
        Method name in ``modin.pandas`` module to refer to.

    Returns
    -------
    callable
    """
    note = _refer_to_note.format(method)
    return append_to_docstring(note)