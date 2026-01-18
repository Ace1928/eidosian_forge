import collections
import copy
import os
import re
import shlex
from typing import List, Tuple
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
def aot_compile_cpu_meta_graph_def(checkpoint_path, meta_graph_def, output_prefix, signature_def_key, cpp_class, target_triple, target_cpu, variables_to_feed=(), multithreading=False):
    """Compile a `MetaGraphDef` to header+object files in `output_prefix`.

  Use XLA AOT (`tfcompile`) to convert the given meta graph and
  signature into a header + object files.  Also create an include makefile
  that helps identify the appropriate necessary include and library paths
  to incorporate these files into your C++ program.

  Freezing a graph entails restoring the checkpoint and replacing any inputs and
  variables with constants. If values are feed, those are used, else inputs are
  replaced with default all-zero constants. Finally, the graph is pruned and
  then optimized with grappler.

  If the `freeze_graph` is `True`, all variables are embedded as constants
  into the graph and binary objects.  If it is `False`, then the variable
  values become inputs and outputs of the compiled class and the C++
  caller must set these values manually.

  Args:
    checkpoint_path: Python string.  Path to checkpoints/variables.
    meta_graph_def: Instance of `MetaGraphDef`.
    output_prefix: Python string.  Path prefix for outputs.
    signature_def_key: String, the signature_def to use in the SavedModel.
    cpp_class: String, Name of output C++ class.
    target_triple: String, LLVM target triple.
    target_cpu: String, LLVM target cpu name.
    variables_to_feed: A list of strings, the variables that will be fed by the
      user; these won't be frozen.  If `None`, then we will extract all the
      variables in the graph and mark them as to-feed.  The default behavior is
      an empty tuple: all variables must be frozen.
    multithreading: Whether to enable multithreading in the compiled
      computation.  Note that if using this option, the resulting object files
      may have external dependencies on multithreading libraries like nsync.

  Raises:
    RuntimeError: If tensorflow was not built with XLA.
    ImportError: If tensorflow was built with XLA but there was another
      issue importing the tfcompile python wrapper.
    ValueError: If `meta_graph_def.signature_def[signature_def_key]` is
      missing or has empty outputs.
  """
    if _pywrap_tfcompile_import_error:
        raise _pywrap_tfcompile_import_error
    else:
        xla_flags = os.environ.get('XLA_FLAGS')
        if not xla_flags:
            xla_flags = '--xla_cpu_multi_thread_eigen={}'.format('true' if multithreading else 'false')
        else:
            xla_flags += ' --xla_cpu_multi_thread_eigen={}'.format('true' if multithreading else 'false')
        os.environ['XLA_FLAGS'] = xla_flags
    temp_dir = test.get_temp_dir()
    file_io.recursive_create_dir(temp_dir)
    frozen_graph_def_location, config_pbtxt_location = freeze_model(checkpoint_path=checkpoint_path, meta_graph_def=meta_graph_def, output_prefix=temp_dir, signature_def_key=signature_def_key, variables_to_feed=variables_to_feed)
    output_dir = os.path.dirname(output_prefix)
    file_io.recursive_create_dir(output_dir)
    entry_point = re.sub('[^0-9a-zA-Z]+', '_', '__xla_' + output_prefix + '__' + cpp_class)
    logging.info('Generating XLA AOT artifacts in: {}'.format(output_dir))
    makefile_inc_location = '{}_makefile.inc'.format(output_prefix)
    with file_io.FileIO(makefile_inc_location, mode='w') as makefile_writer:
        makefile_writer.write(_xla_makefile_string(output_prefix))
    output_prefix = _shlex_quote(output_prefix)
    _pywrap_tfcompile.Compile(graph=frozen_graph_def_location, config=config_pbtxt_location, cpp_class=cpp_class, target_triple=target_triple, target_cpu=target_cpu, entry_point=entry_point, out_function_object='{}.o'.format(output_prefix), out_header='{}.h'.format(output_prefix), out_metadata_object='{}_metadata.o'.format(output_prefix), gen_name_to_index=True, gen_program_shape=False)