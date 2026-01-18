import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def comparisons(self, fileids=None):
    """
        Return all comparisons in the corpus.

        :param fileids: a list or regexp specifying the ids of the files whose
            comparisons have to be returned.
        :return: the given file(s) as a list of Comparison objects.
        :rtype: list(Comparison)
        """
    if fileids is None:
        fileids = self._fileids
    elif isinstance(fileids, str):
        fileids = [fileids]
    return concat([self.CorpusView(path, self._read_comparison_block, encoding=enc) for path, enc, fileid in self.abspaths(fileids, True, True)])