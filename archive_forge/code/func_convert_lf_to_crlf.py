from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
def convert_lf_to_crlf(text_hunk):
    """Convert LF in text hunk into CRLF.

    Args:
      text_hunk: A bytes string representing a text hunk
    Returns: The text hunk with the same type, with LF replaced into CRLF
    """
    intermediary = text_hunk.replace(CRLF, LF)
    return intermediary.replace(LF, CRLF)