from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
@classmethod
def install(cls):
    if cls._replaced_summary_writer:
        raise ValueError('FakeSummaryWriter already installed.')
    cls._replaced_summary_writer = writer.FileWriter
    writer.FileWriter = FakeSummaryWriter
    writer_cache.FileWriter = FakeSummaryWriter