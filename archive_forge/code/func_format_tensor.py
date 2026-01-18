import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
def format_tensor(tensor, tensor_label, include_metadata=False, auxiliary_message=None, include_numeric_summary=False, np_printoptions=None, highlight_options=None):
    """Generate a RichTextLines object showing a tensor in formatted style.

  Args:
    tensor: The tensor to be displayed, as a numpy ndarray or other
      appropriate format (e.g., None representing uninitialized tensors).
    tensor_label: A label for the tensor, as a string. If set to None, will
      suppress the tensor name line in the return value.
    include_metadata: Whether metadata such as dtype and shape are to be
      included in the formatted text.
    auxiliary_message: An auxiliary message to display under the tensor label,
      dtype and shape information lines.
    include_numeric_summary: Whether a text summary of the numeric values (if
      applicable) will be included.
    np_printoptions: A dictionary of keyword arguments that are passed to a
      call of np.set_printoptions() to set the text format for display numpy
      ndarrays.
    highlight_options: (HighlightOptions) options for highlighting elements
      of the tensor.

  Returns:
    A RichTextLines object. Its annotation field has line-by-line markups to
    indicate which indices in the array the first element of each line
    corresponds to.
  """
    lines = []
    font_attr_segs = {}
    if tensor_label is not None:
        lines.append('Tensor "%s":' % tensor_label)
        suffix = tensor_label.split(':')[-1]
        if suffix.isdigit():
            font_attr_segs[0] = [(8, 8 + len(tensor_label), 'bold')]
        else:
            debug_op_len = len(suffix)
            proper_len = len(tensor_label) - debug_op_len - 1
            font_attr_segs[0] = [(8, 8 + proper_len, 'bold'), (8 + proper_len + 1, 8 + proper_len + 1 + debug_op_len, 'yellow')]
    if isinstance(tensor, debug_data.InconvertibleTensorProto):
        if lines:
            lines.append('')
        lines.extend(str(tensor).split('\n'))
        return debugger_cli_common.RichTextLines(lines)
    elif not isinstance(tensor, np.ndarray):
        if lines:
            lines.append('')
        lines.extend(repr(tensor).split('\n'))
        return debugger_cli_common.RichTextLines(lines)
    if include_metadata:
        lines.append('  dtype: %s' % str(tensor.dtype))
        lines.append('  shape: %s' % str(tensor.shape).replace('L', ''))
    if lines:
        lines.append('')
    formatted = debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)
    if auxiliary_message:
        formatted.extend(auxiliary_message)
    if include_numeric_summary:
        formatted.append('Numeric summary:')
        formatted.extend(numeric_summary(tensor))
        formatted.append('')
    if np_printoptions is not None:
        np.set_printoptions(**np_printoptions)
    array_lines = repr(tensor).split('\n')
    if tensor.dtype.type is not np.string_:
        annotations = _annotate_ndarray_lines(array_lines, tensor, np_printoptions=np_printoptions)
    else:
        annotations = None
    formatted_array = debugger_cli_common.RichTextLines(array_lines, annotations=annotations)
    formatted.extend(formatted_array)
    if highlight_options is not None:
        indices_list = list(np.argwhere(highlight_options.criterion(tensor)))
        total_elements = np.size(tensor)
        highlight_summary = 'Highlighted%s: %d of %d element(s) (%.2f%%)' % ('(%s)' % highlight_options.description if highlight_options.description else '', len(indices_list), total_elements, len(indices_list) / float(total_elements) * 100.0)
        formatted.lines[0] += ' ' + highlight_summary
        if indices_list:
            indices_list = [list(indices) for indices in indices_list]
            are_omitted, rows, start_cols, end_cols = locate_tensor_element(formatted, indices_list)
            for is_omitted, row, start_col, end_col in zip(are_omitted, rows, start_cols, end_cols):
                if is_omitted or start_col is None or end_col is None:
                    continue
                if row in formatted.font_attr_segs:
                    formatted.font_attr_segs[row].append((start_col, end_col, highlight_options.font_attr))
                else:
                    formatted.font_attr_segs[row] = [(start_col, end_col, highlight_options.font_attr)]
    return formatted