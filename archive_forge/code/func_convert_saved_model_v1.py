from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export
@tf_export('mlir.experimental.convert_saved_model_v1')
def convert_saved_model_v1(saved_model_path, exported_names, tags, lift_variables, include_variables_in_initializers, upgrade_legacy=True, show_debug_info=False):
    """Converts a v1 SavedModel to MLIR module.

  Args:
    saved_model_path: Path to SavedModel.
    exported_names: Names to export.
    tags: MetaGraphDef to be loaded is identified by the supplied tags.
    lift_variables: Whether to promote tf.VarHandleOp to resource arguments.
    include_variables_in_initializers: Keeps the variables in initializers
      before lifting variables.
    upgrade_legacy: Functionalize the input graph before importing.
    show_debug_info: Whether to include locations in the emitted textual form.

  Returns:
    A textual representation of the MLIR module corresponding to the
    SavedModule.
  """
    return pywrap_mlir.experimental_convert_saved_model_v1(saved_model_path, exported_names, tags, lift_variables, include_variables_in_initializers, upgrade_legacy, show_debug_info)