from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from concurrent import futures
import threading
import time
import uuid
from absl.testing import parameterized
import numpy as np
from seed_rl.grpc.python import ops
from six.moves import range
import tensorflow as tf
def test_no_batching_when_output_rank0(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([2], tf.int32)])
    def foo(unused_x, unused_y):
        return 1
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'Expects arg\\[0\\] to have shape with 1 dimension\\(s\\), but had shape \\[\\]'):
        client.foo(1, 1)