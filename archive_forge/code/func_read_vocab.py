from typing import Optional
import gzip
import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
@classmethod
def read_vocab(cls, vocab_path):
    d_vocab = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 2:
                raise ValueError('vocab file (%s) corrupted. Line (%s)' % (repr(line), vocab_path))
            else:
                wid, word = fields
                d_vocab[wid] = word
    return d_vocab