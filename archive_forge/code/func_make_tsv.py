import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
def make_tsv(metadata, save_path, metadata_header=None):
    if not metadata_header:
        metadata = [str(x) for x in metadata]
    else:
        assert len(metadata_header) == len(metadata[0]), 'len of header must be equal to the number of columns in metadata'
        metadata = ['\t'.join((str(e) for e in l)) for l in [metadata_header] + metadata]
    metadata_bytes = tf.compat.as_bytes('\n'.join(metadata) + '\n')
    with tf.io.gfile.GFile(_gfile_join(save_path, 'metadata.tsv'), 'wb') as f:
        f.write(metadata_bytes)