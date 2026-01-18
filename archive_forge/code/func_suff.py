from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def suff(self, token):
    """
        remove suffixes from the word's end.
        """
    if token.endswith('ك') and len(token) > 3:
        return token[:-1]
    if len(token) > 4:
        for s2 in self.su2:
            if token.endswith(s2):
                return token[:-2]
    if len(token) > 5:
        for s3 in self.su3:
            if token.endswith(s3):
                return token[:-3]
    if token.endswith('ه') and len(token) > 3:
        token = token[:-1]
        return token
    if len(token) > 4:
        for s2 in self.su22:
            if token.endswith(s2):
                return token[:-2]
    if len(token) > 5:
        for s3 in self.su32:
            if token.endswith(s3):
                return token[:-3]
    if token.endswith('نا') and len(token) > 4:
        return token[:-2]
    return token