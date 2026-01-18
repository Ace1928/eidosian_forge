import nltk
from nltk.corpus.reader.api import *
def parsed_docs(self, fileids=None):
    return concat([StreamBackedCorpusView(fileid, self._read_parsed_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])