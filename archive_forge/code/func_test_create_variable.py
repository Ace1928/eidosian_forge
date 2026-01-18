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
def test_create_variable(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    state = [None]

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        if state[0] is None:
            with tf.device('/device:CPU:0'):
                state[0] = tf.Variable(42)
        with tf.device('/device:CPU:0'):
            return x + state[0]
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    self.assertAllEqual(42, state[0].read_value())
    self.assertAllEqual(43, client.foo(1))
    state[0].assign(0)
    self.assertAllEqual(1, client.foo(1))
    server.shutdown()