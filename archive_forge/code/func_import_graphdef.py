from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *
def import_graphdef(graphdef, pass_pipeline, show_debug_info, input_names=None, input_data_types=None, input_data_shapes=None, output_names=[]):
    if input_names is not None:
        return ImportGraphDef(str(graphdef).encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info, ','.join(input_names).encode('utf-8'), ','.join(input_data_types).encode('utf-8'), ':'.join(input_data_shapes).encode('utf-8'), ','.join(output_names).encode('utf-8'))
    return ImportGraphDef(str(graphdef).encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info)