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
def test_bind_multiple_functions(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        return x + 1

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)])
    def bar(x, y):
        return x * y
    server.bind(foo)
    server.bind(bar)
    server.start()
    client = ops.Client(address)
    self.assertAllEqual(43, client.foo(42))
    self.assertAllEqual(100, client.bar(10, 10))
    server.shutdown()