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
def test_nests(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    signature = (tf.TensorSpec([], tf.int32, name='arg1'), Some(tf.TensorSpec([], tf.int32, name='arg2'), [tf.TensorSpec([], tf.int32, name='arg3'), tf.TensorSpec([], tf.int32, name='arg4')]))

    @tf.function(input_signature=signature)
    def foo(*args):
        return tf.nest.map_structure(lambda t: t + 1, args)
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    inputs = (1, Some(2, [3, 4]))
    expected_outputs = (2, Some(3, [4, 5]))
    outputs = client.foo(inputs)
    outputs = tf.nest.map_structure(lambda t: t.numpy(), outputs)
    tf.nest.assert_same_structure(expected_outputs, outputs)
    self.assertAllEqual(tf.nest.flatten(expected_outputs), tf.nest.flatten(outputs))
    server.shutdown()