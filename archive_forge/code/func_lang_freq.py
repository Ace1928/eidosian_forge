import re
from os import path
from nltk.corpus.reader import CorpusReader
from nltk.data import ZipFilePathPointer
from nltk.probability import FreqDist
def lang_freq(self, lang):
    """Return n-gram FreqDist for a specific language
        given ISO 639-3 language code"""
    if lang not in self._all_lang_freq:
        self._all_lang_freq[lang] = self._load_lang_ngrams(lang)
    return self._all_lang_freq[lang]