import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
def remove_choice(self, segm):
    ret = []
    prev_txt_end = -1
    prev_txt_nr = -1
    for word in segm:
        txt_nr = self.get_segm_id(word)
        if self.get_sent_beg(word) > prev_txt_end - 1 or prev_txt_nr != txt_nr:
            ret.append(word)
            prev_txt_end = self.get_sent_end(word)
        prev_txt_nr = txt_nr
    return ret