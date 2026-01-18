import re
from os import path
from nltk.corpus.reader import CorpusReader
from nltk.data import ZipFilePathPointer
from nltk.probability import FreqDist
def crubadan_to_iso(self, lang):
    """Return ISO 639-3 code given internal Crubadan code"""
    for i in self._lang_mapping_data:
        if i[0].lower() == lang.lower():
            return i[1]