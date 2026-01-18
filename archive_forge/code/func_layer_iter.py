import re
from google.protobuf import text_format # pylint: disable=relative-import
def layer_iter(layers, layer_names):
    """Iterate over all layers"""
    if use_caffe:
        for layer_idx, layer in enumerate(layers):
            layer_name = re.sub('[-/]', '_', layer_names[layer_idx])
            layer_type = layer.type
            layer_blobs = layer.blobs
            yield (layer_name, layer_type, layer_blobs)
    else:
        for layer in layers:
            layer_name = re.sub('[-/]', '_', layer.name)
            layer_type = layer.type
            layer_blobs = layer.blobs
            yield (layer_name, layer_type, layer_blobs)