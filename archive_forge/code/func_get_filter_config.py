from typing import Callable, Optional, Union
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
@DeveloperAPI
def get_filter_config(shape):
    """Returns a default Conv2D filter config (list) for a given image shape.

    Args:
        shape (Tuple[int]): The input (image) shape, e.g. (84,84,3).

    Returns:
        List[list]: The Conv2D filter configuration usable as `conv_filters`
            inside a model config dict.
    """
    filters_480x640 = [[16, [24, 32], [14, 18]], [32, [6, 6], 4], [256, [9, 9], 1]]
    filters_240x320 = [[16, [12, 16], [7, 9]], [32, [6, 6], 4], [256, [9, 9], 1]]
    filters_96x96 = [[16, [8, 8], 4], [32, [4, 4], 2], [256, [11, 11], 2]]
    filters_84x84 = [[16, [8, 8], 4], [32, [4, 4], 2], [256, [11, 11], 1]]
    filters_64x64 = [[32, [4, 4], 2], [64, [4, 4], 2], [128, [4, 4], 2], [256, [4, 4], 2]]
    filters_42x42 = [[16, [4, 4], 2], [32, [4, 4], 2], [256, [11, 11], 1]]
    filters_10x10 = [[16, [5, 5], 2], [32, [5, 5], 2]]
    shape = list(shape)
    if len(shape) in [2, 3] and (shape[:2] == [480, 640] or shape[1:] == [480, 640]):
        return filters_480x640
    elif len(shape) in [2, 3] and (shape[:2] == [240, 320] or shape[1:] == [240, 320]):
        return filters_240x320
    elif len(shape) in [2, 3] and (shape[:2] == [96, 96] or shape[1:] == [96, 96]):
        return filters_96x96
    elif len(shape) in [2, 3] and (shape[:2] == [84, 84] or shape[1:] == [84, 84]):
        return filters_84x84
    elif len(shape) in [2, 3] and (shape[:2] == [64, 64] or shape[1:] == [64, 64]):
        return filters_64x64
    elif len(shape) in [2, 3] and (shape[:2] == [42, 42] or shape[1:] == [42, 42]):
        return filters_42x42
    elif len(shape) in [2, 3] and (shape[:2] == [10, 10] or shape[1:] == [10, 10]):
        return filters_10x10
    else:
        raise ValueError('No default configuration for obs shape {}'.format(shape) + ', you must specify `conv_filters` manually as a model option. Default configurations are only available for inputs of the following shapes: [42, 42, K], [84, 84, K], [64, 64, K], [10, 10, K], [240, 320, K], and [480, 640, K]. You may alternatively want to use a custom model or preprocessor.')