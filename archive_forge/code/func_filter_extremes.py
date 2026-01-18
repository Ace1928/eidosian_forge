import logging
import itertools
import zlib
from gensim import utils
def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000):
    """Filter tokens in the debug dictionary by their frequency.

        Since :class:`~gensim.corpora.hashdictionary.HashDictionary` id range is fixed and doesn't depend on the number
        of tokens seen, this doesn't really "remove" anything. It only clears some
        internal corpus statistics, for easier debugging and a smaller RAM footprint.

        Warnings
        --------
        Only makes sense when `debug=True`.

        Parameters
        ----------
        no_below : int, optional
            Keep tokens which are contained in at least `no_below` documents.
        no_above : float, optional
            Keep tokens which are contained in no more than `no_above` documents
            (fraction of total corpus size, not an absolute number).
        keep_n : int, optional
            Keep only the first `keep_n` most frequent tokens.

        Notes
        -----
        For tokens that appear in:

        #. Less than `no_below` documents (absolute number) or 

        #. More than `no_above` documents (fraction of total corpus size, **not absolute number**).
        #. After (1) and (2), keep only the first `keep_n` most frequent tokens (or keep all if `None`).

        """
    no_above_abs = int(no_above * self.num_docs)
    ok = [item for item in self.dfs_debug.items() if no_below <= item[1] <= no_above_abs]
    ok = frozenset((word for word, freq in sorted(ok, key=lambda x: -x[1])[:keep_n]))
    self.dfs_debug = {word: freq for word, freq in self.dfs_debug.items() if word in ok}
    self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if token in self.dfs_debug}
    self.id2token = {tokenid: {token for token in tokens if token in self.dfs_debug} for tokenid, tokens in self.id2token.items()}
    self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if self.id2token.get(tokenid, False)}
    logger.info('kept statistics for which were in no less than %i and no more than %i (=%.1f%%) documents', no_below, no_above_abs, 100.0 * no_above)