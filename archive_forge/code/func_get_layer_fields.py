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
def get_layer_fields(layer, prefix=''):
    output_shape = format_layer_shape(layer)
    name = prefix + layer.name
    cls_name = layer.__class__.__name__
    name = rich.markup.escape(name)
    name += f' ({highlight_symbol(rich.markup.escape(cls_name))})'
    if not hasattr(layer, 'built'):
        params = highlight_number(0)
    elif not layer.built:
        params = highlight_number(0) + ' (unbuilt)'
    else:
        params = highlight_number(f'{layer.count_params():,}')
    fields = [name, output_shape, params]
    if not sequential_like:
        fields.append(get_connections(layer))
    if show_trainable:
        if layer.weights:
            fields.append(bold_text('Y', color=34) if layer.trainable else bold_text('N', color=9))
        else:
            fields.append(bold_text('-'))
    return fields