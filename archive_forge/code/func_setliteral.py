import _markupbase
import re
def setliteral(self, *args):
    """Enter literal mode (CDATA).

        Intended for derived classes only.
        """
    self.literal = 1