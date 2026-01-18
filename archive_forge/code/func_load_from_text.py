from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple
from gensim import utils
@staticmethod
def load_from_text(fname):
    """Load a previously stored :class:`~gensim.corpora.dictionary.Dictionary` from a text file.

        Mirror function to :meth:`~gensim.corpora.dictionary.Dictionary.save_as_text`.

        Parameters
        ----------
        fname: str
            Path to a file produced by :meth:`~gensim.corpora.dictionary.Dictionary.save_as_text`.

        See Also
        --------
        :meth:`~gensim.corpora.dictionary.Dictionary.save_as_text`
            Save :class:`~gensim.corpora.dictionary.Dictionary` to text file.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import Dictionary
            >>> from gensim.test.utils import get_tmpfile
            >>>
            >>> tmp_fname = get_tmpfile("dictionary")
            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
            >>>
            >>> dct = Dictionary(corpus)
            >>> dct.save_as_text(tmp_fname)
            >>>
            >>> loaded_dct = Dictionary.load_from_text(tmp_fname)
            >>> assert dct.token2id == loaded_dct.token2id

        """
    result = Dictionary()
    with utils.open(fname, 'rb') as f:
        for lineno, line in enumerate(f):
            line = utils.to_unicode(line)
            if lineno == 0:
                if line.strip().isdigit():
                    result.num_docs = int(line.strip())
                    continue
                else:
                    logging.warning('Text does not contain num_docs on the first line.')
            try:
                wordid, word, docfreq = line[:-1].split('\t')
            except Exception:
                raise ValueError('invalid line in dictionary file %s: %s' % (fname, line.strip()))
            wordid = int(wordid)
            if word in result.token2id:
                raise KeyError('token %s is defined as ID %d and as ID %d' % (word, wordid, result.token2id[word]))
            result.token2id[word] = wordid
            result.dfs[wordid] = int(docfreq)
    return result