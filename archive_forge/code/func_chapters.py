import nltk.data
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
def chapters(self, fileids=None):
    """
        :return: the given file(s) as a list of
            chapters, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
    return concat([self.CorpusView(fileid, self._read_para_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])