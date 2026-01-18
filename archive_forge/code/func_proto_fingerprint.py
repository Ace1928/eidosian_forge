import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def proto_fingerprint(message_proto):
    serialized_message = message_proto.SerializeToString()
    hasher = hashlib.sha256(serialized_message)
    return hasher.hexdigest()