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
def test_client_side_batching(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([4], tf.int32)])
    def foo(x):
        return x + 1
    server.bind(foo)
    server.start()

    def client(x):
        client = ops.Client(address)
        return client.foo(x)
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(client, np.array([42, 43], np.int32))
        f2 = executor.submit(client, np.array([142, 143], np.int32))
        self.assertAllEqual(f1.result(), [43, 44])
        self.assertAllEqual(f2.result(), [143, 144])
    server.shutdown()