import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
@PublicAPI
def flatten_inputs_to_1d_tensor(inputs: TensorStructType, spaces_struct: Optional[SpaceStruct]=None, time_axis: bool=False) -> TensorType:
    """Flattens arbitrary input structs according to the given spaces struct.

    Returns a single 1D tensor resulting from the different input
    components' values.

    Thereby:
    - Boxes (any shape) get flattened to (B, [T]?, -1). Note that image boxes
    are not treated differently from other types of Boxes and get
    flattened as well.
    - Discrete (int) values are one-hot'd, e.g. a batch of [1, 0, 3] (B=3 with
    Discrete(4) space) results in [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]].
    - MultiDiscrete values are multi-one-hot'd, e.g. a batch of
    [[0, 2], [1, 4]] (B=2 with MultiDiscrete([2, 5]) space) results in
    [[1, 0,  0, 0, 1, 0, 0], [0, 1,  0, 0, 0, 0, 1]].

    Args:
        inputs: The inputs to be flattened.
        spaces_struct: The structure of the spaces that behind the input
        time_axis: Whether all inputs have a time-axis (after the batch axis).
            If True, will keep not only the batch axis (0th), but the time axis
            (1st) as-is and flatten everything from the 2nd axis up.

    Returns:
        A single 1D tensor resulting from concatenating all
        flattened/one-hot'd input components. Depending on the time_axis flag,
        the shape is (B, n) or (B, T, n).

    .. testcode::
        :skipif: True

        # B=2
        from ray.rllib.utils.tf_utils import flatten_inputs_to_1d_tensor
        from gymnasium.spaces import Discrete, Box
        out = flatten_inputs_to_1d_tensor(
            {"a": [1, 0], "b": [[[0.0], [0.1]], [1.0], [1.1]]},
            spaces_struct=dict(a=Discrete(2), b=Box(shape=(2, 1)))
        )
        print(out)

        # B=2; T=2
        out = flatten_inputs_to_1d_tensor(
            ([[1, 0], [0, 1]],
             [[[0.0, 0.1], [1.0, 1.1]], [[2.0, 2.1], [3.0, 3.1]]]),
            spaces_struct=tuple([Discrete(2), Box(shape=(2, ))]),
            time_axis=True
        )
        print(out)

    .. testoutput::

        [[0.0, 1.0,  0.0, 0.1], [1.0, 0.0,  1.0, 1.1]]  # B=2 n=4
        [[[0.0, 1.0, 0.0, 0.1], [1.0, 0.0, 1.0, 1.1]],
        [[1.0, 0.0, 2.0, 2.1], [0.0, 1.0, 3.0, 3.1]]]  # B=2 T=2 n=4
    """
    flat_inputs = tree.flatten(inputs)
    flat_spaces = tree.flatten(spaces_struct) if spaces_struct is not None else [None] * len(flat_inputs)
    B = None
    T = None
    out = []
    for input_, space in zip(flat_inputs, flat_spaces):
        input_ = tf.convert_to_tensor(input_)
        shape = tf.shape(input_)
        if B is None:
            B = shape[0]
            if time_axis:
                T = shape[1]
        if isinstance(space, Discrete):
            if time_axis:
                input_ = tf.reshape(input_, [B * T])
            out.append(tf.cast(one_hot(input_, space), tf.float32))
        elif isinstance(space, MultiDiscrete):
            if time_axis:
                input_ = tf.reshape(input_, [B * T, -1])
            out.append(tf.cast(one_hot(input_, space), tf.float32))
        else:
            if time_axis:
                input_ = tf.reshape(input_, [B * T, -1])
            else:
                input_ = tf.reshape(input_, [B, -1])
            out.append(tf.cast(input_, tf.float32))
    merged = tf.concat(out, axis=-1)
    if time_axis:
        merged = tf.reshape(merged, [B, T, -1])
    return merged