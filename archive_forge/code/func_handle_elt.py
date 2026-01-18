from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader, XMLCorpusView
def handle_elt(self, elt, context):
    if self._sent:
        return self.handle_sent(elt)
    else:
        return self.handle_word(elt)