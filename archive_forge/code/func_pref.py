from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def pref(self, token):
    """
        remove prefixes from the words' beginning.
        """
    if len(token) > 5:
        for p3 in self.pr3:
            if token.startswith(p3):
                return token[3:]
    if len(token) > 6:
        for p4 in self.pr4:
            if token.startswith(p4):
                return token[4:]
    if len(token) > 5:
        for p3 in self.pr32:
            if token.startswith(p3):
                return token[3:]
    if len(token) > 4:
        for p2 in self.pr2:
            if token.startswith(p2):
                return token[2:]