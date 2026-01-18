import os
import sys
from keras.src.api_export import keras_export
from keras.src.utils import io_utils
def make_layer_label(layer, **kwargs):
    class_name = layer.__class__.__name__
    show_layer_names = kwargs.pop('show_layer_names')
    show_layer_activations = kwargs.pop('show_layer_activations')
    show_dtype = kwargs.pop('show_dtype')
    show_shapes = kwargs.pop('show_shapes')
    show_trainable = kwargs.pop('show_trainable')
    if kwargs:
        raise ValueError(f'Invalid kwargs: {kwargs}')
    table = '<<table border="0" cellborder="1" bgcolor="black" cellpadding="10">'
    colspan = max(1, sum((int(x) for x in (show_dtype, show_shapes, show_trainable))))
    if show_layer_names:
        table += f'<tr><td colspan="{colspan}" bgcolor="black"><font point-size="16" color="white"><b>{layer.name}</b> ({class_name})</font></td></tr>'
    else:
        table += f'<tr><td colspan="{colspan}" bgcolor="black"><font point-size="16" color="white"><b>{class_name}</b></font></td></tr>'
    if show_layer_activations and hasattr(layer, 'activation') and (layer.activation is not None):
        table += f'<tr><td bgcolor="white" colspan="{colspan}"><font point-size="14">Activation: <b>{get_layer_activation_name(layer)}</b></font></td></tr>'
    cols = []
    if show_shapes:
        shape = None
        try:
            shape = layer.output.shape
        except ValueError:
            pass
        cols.append(f'<td bgcolor="white"><font point-size="14">Output shape: <b>{shape or '?'}</b></font></td>')
    if show_dtype:
        dtype = None
        try:
            dtype = layer.output.dtype
        except ValueError:
            pass
        cols.append(f'<td bgcolor="white"><font point-size="14">Output dtype: <b>{dtype or '?'}</b></font></td>')
    if show_trainable and hasattr(layer, 'trainable') and layer.weights:
        if layer.trainable:
            cols.append('<td bgcolor="forestgreen"><font point-size="14" color="white"><b>Trainable</b></font></td>')
        else:
            cols.append('<td bgcolor="firebrick"><font point-size="14" color="white"><b>Non-trainable</b></font></td>')
    if cols:
        colspan = len(cols)
    else:
        colspan = 1
    if cols:
        table += '<tr>' + ''.join(cols) + '</tr>'
    table += '</table>>'
    return table