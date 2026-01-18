import collections
import numpy as np
import tensorflow.compat.v2 as tf
def assert_extracted_output_equal(self, combiner, acc1, acc2, msg=None):
    data_1 = combiner.extract(acc1)
    data_2 = combiner.extract(acc2)
    self.assertAllCloseOrEqual(data_1, data_2, msg=msg)