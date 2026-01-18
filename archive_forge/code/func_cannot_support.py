import warnings
from ...utils.import_utils import check_if_transformers_greater
from .decoder_models import (
from .encoder_models import (
@staticmethod
def cannot_support(model_type: str) -> bool:
    """
        Returns True if a given model type can not be supported by PyTorch's Better Transformer.

        Args:
            model_type (`str`):
                The model type to check.
        """
    return model_type in BetterTransformerManager.CAN_NOT_BE_SUPPORTED