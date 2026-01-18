import os
from typing import Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ... import AutoTokenizer
from ...utils import logging
def convert_tfrecord_to_np(block_records_path: str, num_block_records: int) -> np.ndarray:
    import tensorflow.compat.v1 as tf
    blocks_dataset = tf.data.TFRecordDataset(block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(num_block_records, drop_remainder=True)
    np_record = next(blocks_dataset.take(1).as_numpy_iterator())
    return np_record