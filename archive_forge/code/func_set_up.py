from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow.compat.v2 as tf
from absl import flags
from absl.testing import absltest
from keras.src.testing_infra import keras_doctest_lib
import doctest  # noqa: E402
def set_up(self, _):
    self.setUp()