from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def fem2masc(self, token):
    """
        transform the word from the feminine form to the masculine form.
        """
    if token.endswith('ة') and len(token) > 3:
        return token[:-1]