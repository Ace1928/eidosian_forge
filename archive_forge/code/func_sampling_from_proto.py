import math
from keras_tuner.src import protos
def sampling_from_proto(sampling):
    if sampling == protos.get_proto().Sampling.LINEAR:
        return 'linear'
    if sampling == protos.get_proto().Sampling.LOG:
        return 'log'
    if sampling == protos.get_proto().Sampling.REVERSE_LOG:
        return 'reverse_log'
    raise ValueError(f"Expected sampling to be one of predefined proto values. Received: '{sampling}'.")