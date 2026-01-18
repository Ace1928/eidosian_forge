import enum
import warnings
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
def check_inputs(self, input_length: int, min_length: int, max_length: int):
    if input_length > 0.9 * max_length:
        logger.warning(f"Your input_length: {input_length} is bigger than 0.9 * max_length: {max_length}. You might consider increasing your max_length manually, e.g. translator('...', max_length=400)")
    return True