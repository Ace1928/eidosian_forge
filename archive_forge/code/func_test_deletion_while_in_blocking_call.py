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
def test_deletion_while_in_blocking_call(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    q = tf.queue.FIFOQueue(1, [tf.int32])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        q.enqueue(x)
        q.enqueue(x)
        q.enqueue(x)
        return x
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(client.foo, 42)
        q.dequeue()
        del server
        try:
            f.result()
        except tf.errors.CancelledError:
            pass
        except tf.errors.UnavailableError:
            pass