from sys import maxsize
from nltk.util import trigrams
def calc_dist(self, lang, trigram, text_profile):
    """Calculate the "out-of-place" measure between the
        text and language profile for a single trigram"""
    lang_fd = self._corpus.lang_freq(lang)
    dist = 0
    if trigram in lang_fd:
        idx_lang_profile = list(lang_fd.keys()).index(trigram)
        idx_text = list(text_profile.keys()).index(trigram)
        dist = abs(idx_lang_profile - idx_text)
    else:
        dist = maxsize
    return dist