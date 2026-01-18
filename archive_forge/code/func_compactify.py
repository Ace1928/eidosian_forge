from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple
from gensim import utils
def compactify(self):
    """Assign new word ids to all words, shrinking any gaps."""
    logger.debug('rebuilding dictionary, shrinking gaps')
    idmap = dict(zip(sorted(self.token2id.values()), range(len(self.token2id))))
    self.token2id = {token: idmap[tokenid] for token, tokenid in self.token2id.items()}
    self.id2token = {}
    self.dfs = {idmap[tokenid]: freq for tokenid, freq in self.dfs.items()}
    self.cfs = {idmap[tokenid]: freq for tokenid, freq in self.cfs.items()}