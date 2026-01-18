import io
import re
from nltk.corpus import perluniprops
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import xml_unescape
def lang_independent_sub(self, text):
    """Performs the language independent string substituitions."""
    regexp, substitution = self.STRIP_SKIP
    text = regexp.sub(substitution, text)
    text = xml_unescape(text)
    regexp, substitution = self.STRIP_EOL_HYPHEN
    text = regexp.sub(substitution, text)
    return text