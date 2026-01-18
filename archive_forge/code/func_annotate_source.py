import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def annotate_source(dump, source_file_path, do_dumped_tensors=False, file_stack_top=False, min_line=None, max_line=None):
    """Annotate a Python source file with a list of ops created at each line.

  (The annotation doesn't change the source file itself.)

  Args:
    dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
      has been loaded.
    source_file_path: (`str`) Path to the source file being annotated.
    do_dumped_tensors: (`str`) Whether dumped Tensors, instead of ops are to be
      used to annotate the source file.
    file_stack_top: (`bool`) Whether only the top stack trace in the
      specified source file is to be annotated.
    min_line: (`None` or `int`) The 1-based line to start annotate the source
      file from (inclusive).
    max_line: (`None` or `int`) The 1-based line number to end the annotation
      at (exclusive).

  Returns:
    A `dict` mapping 1-based line number to a list of op name(s) created at
      that line, or tensor names if `do_dumped_tensors` is True.

  Raises:
    ValueError: If the dump object does not have a Python graph set.
  """
    py_graph = dump.python_graph
    if not py_graph:
        raise ValueError('Cannot perform source annotation due to a lack of set Python graph in the dump object')
    source_file_path = _norm_abs_path(source_file_path)
    line_to_op_names = {}
    for op in py_graph.get_operations():
        for file_path, line_number, _, _ in reversed(dump.node_traceback(op.name)):
            if min_line is not None and line_number < min_line or (max_line is not None and line_number >= max_line):
                continue
            if _norm_abs_path(file_path) != source_file_path:
                continue
            if do_dumped_tensors:
                watch_keys = dump.debug_watch_keys(op.name)
                items_to_append = list(set(map(_convert_watch_key_to_tensor_name, watch_keys)))
            else:
                items_to_append = [op.name]
            if line_number in line_to_op_names:
                line_to_op_names[line_number].extend(items_to_append)
            else:
                line_to_op_names[line_number] = items_to_append
            if file_stack_top:
                break
    return line_to_op_names