from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def tagged_words(self, fileids=None, tagset=None):
    if tagset and tagset != self._tagset:
        tag_mapping_function = lambda t: map_tag(self._tagset, tagset, t)
    else:
        tag_mapping_function = None
    return concat([IndianCorpusView(fileid, enc, True, False, tag_mapping_function) for fileid, enc in self.abspaths(fileids, True)])