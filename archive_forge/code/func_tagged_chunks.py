from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree import Tree
def tagged_chunks(self, fileids=None, tag='pos' or 'sem' or 'both'):
    """
        :return: the given file(s) as a list of tagged chunks, represented
            in tree form.
        :rtype: list(Tree)

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`
            to indicate the kind of tags to include.  Semantic tags consist of
            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity
            without a specific entry in WordNet.  (Named entities of type 'other'
            have no lemma.  Other chunks not in WordNet have no semantic tag.
            Punctuation tokens have `None` for their part of speech tag.)
        """
    return self._items(fileids, 'chunk', False, tag != 'sem', tag != 'pos')