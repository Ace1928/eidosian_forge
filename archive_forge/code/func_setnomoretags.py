import _markupbase
import re
def setnomoretags(self):
    """Enter literal mode (CDATA) till EOF.

        Intended for derived classes only.
        """
    self.nomoretags = self.literal = 1