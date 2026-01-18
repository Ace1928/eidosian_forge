from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from google.protobuf import text_format
def main(_):
    convert_checkpoint(FLAGS.estimator_type, FLAGS.source_checkpoint, FLAGS.source_graph, FLAGS.target_checkpoint)