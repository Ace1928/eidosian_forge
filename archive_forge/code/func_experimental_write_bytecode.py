from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export
@tf_export('mlir.experimental.write_bytecode')
def experimental_write_bytecode(filename, mlir_txt):
    """Writes an MLIR module out as bytecode.

  Args:
    filename: The filename to write to.
    mlir_txt: The MLIR module in textual format.
  """
    pywrap_mlir.experimental_write_bytecode(filename, mlir_txt)