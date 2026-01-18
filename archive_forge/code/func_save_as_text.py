import logging
import itertools
import zlib
from gensim import utils
def save_as_text(self, fname):
    """Save the debug token=>id mapping to a text file.

        Warnings
        --------
        Only makes sense when `debug=True`, for debugging.

        Parameters
        ----------
        fname : str
            Path to output file.

        Notes
        -----
        The format is:
        `id[TAB]document frequency of this id[TAB]tab-separated set of words in UTF8 that map to this id[NEWLINE]`.


        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import HashDictionary
            >>> from gensim.test.utils import get_tmpfile
            >>>
            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
            >>> data = HashDictionary(corpus)
            >>> data.save_as_text(get_tmpfile("dictionary_in_text_format"))

        """
    logger.info('saving %s mapping to %s' % (self, fname))
    with utils.open(fname, 'wb') as fout:
        for tokenid in self.keys():
            words = sorted(self[tokenid])
            if words:
                words_df = [(word, self.dfs_debug.get(word, 0)) for word in words]
                words_df = ['%s(%i)' % item for item in sorted(words_df, key=lambda x: -x[1])]
                words_df = '\t'.join(words_df)
                fout.write(utils.to_utf8('%i\t%i\t%s\n' % (tokenid, self.dfs.get(tokenid, 0), words_df)))