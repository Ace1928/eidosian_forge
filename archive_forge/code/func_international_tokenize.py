import io
import re
from nltk.corpus import perluniprops
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import xml_unescape
def international_tokenize(self, text, lowercase=False, split_non_ascii=True, return_str=False):
    text = str(text)
    regexp, substitution = self.STRIP_SKIP
    text = regexp.sub(substitution, text)
    regexp, substitution = self.STRIP_EOL_HYPHEN
    text = regexp.sub(substitution, text)
    text = xml_unescape(text)
    if lowercase:
        text = text.lower()
    for regexp, substitution in self.INTERNATIONAL_REGEXES:
        text = regexp.sub(substitution, text)
    text = ' '.join(text.strip().split())
    return text if return_str else text.split()