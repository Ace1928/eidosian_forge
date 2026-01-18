import os
import warnings
from functools import partial
from math import ceil
from uuid import uuid4
import numpy as np
import pyarrow as pa
from multiprocess import get_context
from .. import config
def minimal_tf_collate_fn_with_renaming(features):
    batch = minimal_tf_collate_fn(features)
    if 'label' in batch:
        batch['labels'] = batch['label']
        del batch['label']
    return batch