import copy
import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from .dynamic_module_utils import custom_object_save
from .tokenization_utils_base import PreTrainedTokenizerBase
from .utils import (
@classmethod
def from_args_and_dict(cls, args, processor_dict: Dict[str, Any], **kwargs):
    """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
    processor_dict = processor_dict.copy()
    return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)
    if 'processor_class' in processor_dict:
        del processor_dict['processor_class']
    if 'auto_map' in processor_dict:
        del processor_dict['auto_map']
    processor = cls(*args, **processor_dict)
    for key in set(kwargs.keys()):
        if hasattr(processor, key):
            setattr(processor, key, kwargs.pop(key))
    logger.info(f'Processor {processor}')
    if return_unused_kwargs:
        return (processor, kwargs)
    else:
        return processor