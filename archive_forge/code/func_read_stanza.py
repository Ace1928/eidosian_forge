import re
from .. import osutils
from ..iterablefile import IterableFile
def read_stanza(line_iter):
    """Return new Stanza read from list of lines or a file

    Returns one Stanza that was read, or returns None at end of file.  If a
    blank line follows the stanza, it is consumed.  It's not an error for
    there to be no blank at end of file.  If there is a blank file at the
    start of the input this is really an empty stanza and that is returned.

    Only the stanza lines and the trailing blank (if any) are consumed
    from the line_iter.

    The raw lines must be in utf-8 encoding.
    """
    return _read_stanza_utf8(line_iter)