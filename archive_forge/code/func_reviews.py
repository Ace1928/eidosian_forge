import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def reviews(self, fileids=None):
    """
        Return all the reviews as a list of Review objects. If `fileids` is
        specified, return all the reviews from each of the specified files.

        :param fileids: a list or regexp specifying the ids of the files whose
            reviews have to be returned.
        :return: the given file(s) as a list of reviews.
        """
    if fileids is None:
        fileids = self._fileids
    return concat([self.CorpusView(fileid, self._read_review_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])