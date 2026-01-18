from typing import Optional
import gzip
import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
@classmethod
def read_label2answer(cls, label2answer_path_gz, d_vocab):
    lines = cls.readlines(label2answer_path_gz)
    d_label_answer = {}
    for line in lines:
        fields = line.rstrip('\n').split('\t')
        if len(fields) != 2:
            raise ValueError('label2answer file (%s) corrupted. Line (%s)' % (repr(line), label2answer_path_gz))
        else:
            aid, s_wids = fields
            sent = cls.wids2sent(s_wids.split(), d_vocab)
            d_label_answer[aid] = sent
    return d_label_answer