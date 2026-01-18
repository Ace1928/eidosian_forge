import enum
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
@property
def embedding_feature(self):
    """TPU embedding feature.

    Returns:
      An EmbeddingFeature enum.
    """
    return HardwareFeature._embedding_feature_proto_to_string(self.tpu_hardware_feature_proto.embedding_feature)