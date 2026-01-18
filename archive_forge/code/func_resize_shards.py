from __future__ import print_function
import logging
import os
import math
import time
import numpy
import scipy.sparse as sparse
import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus
def resize_shards(self, shardsize):
    """
        Re-process the dataset to new shard size. This may take pretty long.
        Also, note that you need some space on disk for this one (we're
        assuming there is enough disk space for double the size of the dataset
        and that there is enough memory for old + new shardsize).

        :type shardsize: int
        :param shardsize: The new shard size.

        """
    n_new_shards = int(math.floor(self.n_docs / float(shardsize)))
    if self.n_docs % shardsize != 0:
        n_new_shards += 1
    new_shard_names = []
    new_offsets = [0]
    for new_shard_idx in range(n_new_shards):
        new_start = shardsize * new_shard_idx
        new_stop = new_start + shardsize
        if new_stop > self.n_docs:
            assert new_shard_idx == n_new_shards - 1, 'Shard no. %r that ends at %r over last document (%r) is not the last projected shard (%r)' % (new_shard_idx, new_stop, self.n_docs, n_new_shards)
            new_stop = self.n_docs
        new_shard = self[new_start:new_stop]
        new_shard_name = self._resized_shard_name(new_shard_idx)
        new_shard_names.append(new_shard_name)
        try:
            self.save_shard(new_shard, new_shard_idx, new_shard_name)
        except Exception:
            for new_shard_name in new_shard_names:
                os.remove(new_shard_name)
            raise
        new_offsets.append(new_stop)
    old_shard_names = [self._shard_name(n) for n in range(self.n_shards)]
    try:
        for old_shard_n, old_shard_name in enumerate(old_shard_names):
            os.remove(old_shard_name)
    except Exception as e:
        logger.exception('Error during old shard no. %d removal: %s.\nAttempting to at least move new shards in.', old_shard_n, str(e))
    finally:
        try:
            for shard_n, new_shard_name in enumerate(new_shard_names):
                os.rename(new_shard_name, self._shard_name(shard_n))
        except Exception as e:
            logger.exception(e)
            raise RuntimeError('Resizing completely failed. Sorry, dataset is probably ruined...')
        finally:
            self.n_shards = n_new_shards
            self.offsets = new_offsets
            self.shardsize = shardsize
            self.reset()