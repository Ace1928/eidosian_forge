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
def test_invalid_shape(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.int32)])
    def foo(x):
        return x
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'Expects arg\\[0\\] to have shape with suffix \\[3\\], but had shape \\[3,4\\]'):
        client.foo(tf.zeros([3, 4], tf.int32))
    server.shutdown()