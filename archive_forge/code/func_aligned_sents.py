from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import (
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.translate import AlignedSent, Alignment
def aligned_sents(self, fileids=None):
    """
        :return: the given file(s) as a list of AlignedSent objects.
        :rtype: list(AlignedSent)
        """
    return concat([AlignedSentCorpusView(fileid, enc, True, True, self._word_tokenizer, self._sent_tokenizer, self._alignedsent_block_reader) for fileid, enc in self.abspaths(fileids, True)])