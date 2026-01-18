import re
import warnings
from contextlib import contextmanager
from ...processing_utils import ProcessorMixin
@contextmanager
def as_target_processor(self):
    """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
    warnings.warn('`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your labels by using the argument `text` of the regular `__call__` method (either in the same call as your images inputs, or in a separate call.')
    self._in_target_context_manager = True
    self.current_processor = self.tokenizer
    yield
    self.current_processor = self.image_processor
    self._in_target_context_manager = False