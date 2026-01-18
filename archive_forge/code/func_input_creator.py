import argparse
import os
import ray
from ray import air, tune
from ray.rllib.offline import JsonReader, ShuffledInput, IOContext, InputReader
from ray.tune.registry import get_trainable_cls, register_input
def input_creator(ioctx: IOContext) -> InputReader:
    """
    The input creator method can be used in the input registry or set as the
    config["input"] parameter.

    Args:
        ioctx: use this to access the `input_config` arguments.

    Returns:
        instance of ShuffledInput to work with some offline rl algorithms
    """
    return ShuffledInput(CustomJsonReader(ioctx))