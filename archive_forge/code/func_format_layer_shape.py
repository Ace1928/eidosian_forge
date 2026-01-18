import math
import re
import shutil
import rich
import rich.console
import rich.markup
import rich.table
import tree
from keras.src import backend
from keras.src.utils import dtype_utils
from keras.src.utils import io_utils
def format_layer_shape(layer):
    if not layer._inbound_nodes:
        return '?'

    def format_shape(shape):
        highlighted = [highlight_number(x) for x in shape]
        return '(' + ', '.join(highlighted) + ')'
    for i in range(len(layer._inbound_nodes)):
        outputs = layer._inbound_nodes[i].output_tensors
        output_shapes = tree.map_structure(lambda x: format_shape(x.shape), outputs)
    if len(output_shapes) == 1:
        return output_shapes[0]
    out = str(output_shapes)
    out = out.replace("'", '')
    return out