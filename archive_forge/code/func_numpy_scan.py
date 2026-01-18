import numpy as np
import tree
from keras.src.utils.nest import pack_sequence_as
def numpy_scan(f, init, xs, reverse=False, mask=None):
    states = init
    outputs = []
    if mask is not None:
        x, mask = xs
        x = np.flip(x, axis=0) if reverse else x
        mask = np.flip(mask, axis=0) if reverse else mask
        for each_x, each_mask in zip(x, mask):
            states, output = f(states, (each_x, each_mask))
            outputs.append(output)
    else:
        xs = np.flip(xs, axis=0) if reverse else xs
        for x in xs:
            states, output = f(states, x)
            outputs.append(output)
    outputs = np.array(outputs)
    if reverse:
        outputs = np.flip(outputs, axis=0)
    return (states, outputs)