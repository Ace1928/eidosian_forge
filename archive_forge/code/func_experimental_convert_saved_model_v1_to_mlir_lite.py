from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *
def experimental_convert_saved_model_v1_to_mlir_lite(saved_model_path, exported_names, tags, upgrade_legacy, show_debug_info):
    return ExperimentalConvertSavedModelV1ToMlirLite(str(saved_model_path).encode('utf-8'), str(exported_names).encode('utf-8'), str(tags).encode('utf-8'), upgrade_legacy, show_debug_info)