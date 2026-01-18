import logging
import itertools
import zlib
from gensim import utils
def doc2bow(self, document, allow_update=False, return_missing=False):
    """Convert a sequence of words `document` into the bag-of-words format of `[(word_id, word_count)]`
        (e.g. `[(1, 4), (150, 1), (2005, 2)]`).

        Notes
        -----
        Each word is assumed to be a **tokenized and normalized** string. No further preprocessing
        is done on the words in `document`: you have to apply tokenization, stemming etc before calling this method.

        If `allow_update` or `self.allow_update` is set, then also update the dictionary in the process: update overall
        corpus statistics and document frequencies. For each id appearing in this document, increase its document
        frequency (`self.dfs`) by one.

        Parameters
        ----------
        document : sequence of str
            A sequence of word tokens = **tokenized and normalized** strings.
        allow_update : bool, optional
            Update corpus statistics and if `debug=True`, also the reverse id=>word mapping?
        return_missing : bool, optional
            Not used. Only here for compatibility with the Dictionary class.

        Return
        ------
        list of (int, int)
            Document in Bag-of-words (BoW) format.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import HashDictionary
            >>>
            >>> dct = HashDictionary()
            >>> dct.doc2bow(["this", "is", "máma"])
            [(1721, 1), (5280, 1), (22493, 1)]

        """
    result = {}
    missing = {}
    document = sorted(document)
    for word_norm, group in itertools.groupby(document):
        frequency = len(list(group))
        tokenid = self.restricted_hash(word_norm)
        result[tokenid] = result.get(tokenid, 0) + frequency
        if self.debug:
            self.dfs_debug[word_norm] = self.dfs_debug.get(word_norm, 0) + 1
    if allow_update or self.allow_update:
        self.num_docs += 1
        self.num_pos += len(document)
        self.num_nnz += len(result)
        if self.debug:
            for tokenid in result.keys():
                self.dfs[tokenid] = self.dfs.get(tokenid, 0) + 1
    result = sorted(result.items())
    if return_missing:
        return (result, missing)
    else:
        return result