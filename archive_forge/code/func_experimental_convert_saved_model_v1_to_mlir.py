from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *
def experimental_convert_saved_model_v1_to_mlir(saved_model_path, exported_names, tags, lift_variables, include_variables_in_initializers, upgrade_legacy, show_debug_info):
    return ExperimentalConvertSavedModelV1ToMlir(str(saved_model_path).encode('utf-8'), str(exported_names).encode('utf-8'), str(tags).encode('utf-8'), lift_variables, include_variables_in_initializers, upgrade_legacy, show_debug_info)