from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import IndexedCorpus
@staticmethod
def save_corpus(fname, corpus, id2word=None, labels=False, metadata=False):
    """Save a corpus in the SVMlight format.

        The SVMlight `<target>` class tag is taken from the `labels` array, or set to 0 for all documents
        if `labels` is not supplied.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of iterable of (int, float)
            Corpus in BoW format.
        id2word : dict of (str, str), optional
            Mapping id -> word.
        labels : list or False
            An SVMlight `<target>` class tags or False if not present.
        metadata : bool
            ARGUMENT WILL BE IGNORED.

        Returns
        -------
        list of int
            Offsets for each line in file (in bytes).

        """
    logger.info('converting corpus to SVMlight format: %s', fname)
    if labels is not False:
        labels = list(labels)
    offsets = []
    with utils.open(fname, 'wb') as fout:
        for docno, doc in enumerate(corpus):
            label = labels[docno] if labels else 0
            offsets.append(fout.tell())
            fout.write(utils.to_utf8(SvmLightCorpus.doc2line(doc, label)))
    return offsets