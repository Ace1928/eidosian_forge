import logging
import itertools
import zlib
from gensim import utils
def restricted_hash(self, token):
    """Calculate id of the given token.
        Also keep track of what words were mapped to what ids, if `debug=True` was set in the constructor.

        Parameters
        ----------
        token : str
            Input token.

        Return
        ------
        int
            Hash value of `token`.

        """
    h = self.myhash(utils.to_utf8(token)) % self.id_range
    if self.debug:
        self.token2id[token] = h
        self.id2token.setdefault(h, set()).add(token)
    return h