from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import AddedToken, BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, logging
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)