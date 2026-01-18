from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.util import Index
def read_cmudict_block(stream):
    entries = []
    while len(entries) < 100:
        line = stream.readline()
        if line == '':
            return entries
        pieces = line.split()
        entries.append((pieces[0].lower(), pieces[2:]))
    return entries