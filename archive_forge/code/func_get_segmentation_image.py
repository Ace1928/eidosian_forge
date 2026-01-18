import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def get_segmentation_image(masks: np.ndarray, input_size: Tuple, target_size: Tuple, stuff_equiv_classes, deduplicate=False):
    h, w = input_size
    final_h, final_w = target_size
    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)
    if m_id.shape[-1] == 0:
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        m_id = m_id.argmax(-1).reshape(h, w)
    if deduplicate:
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]
    seg_img = id_to_rgb(m_id)
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img