import re
from nltk.stem.api import StemmerI
def adjective(self, token):
    """
        remove the infixes from adjectives
        """
    if len(token) > 5:
        if token.startswith('ا') and token[-3] == 'ا' and token.endswith('ي'):
            return token[:-3] + token[-2]