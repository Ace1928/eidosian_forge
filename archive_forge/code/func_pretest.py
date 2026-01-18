import copy
import re
import types
from .ucre import build_re
def pretest(self, text):
    """Very quick check, that can give false positives.

        Returns true if link MAY BE can exists. Can be used for speed optimization,
        when you need to check that link NOT exists.

        Args:
            text (str): text to search

        Returns:
            bool: ``True`` if a linkable pattern was found, otherwise it is ``False``.
        """
    if re.search(self.re['pretest'], text, flags=re.IGNORECASE):
        return True
    return False