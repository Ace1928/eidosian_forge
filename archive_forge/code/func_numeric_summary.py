import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
def numeric_summary(tensor):
    """Get a text summary of a numeric tensor.

  This summary is only available for numeric (int*, float*, complex*) and
  Boolean tensors.

  Args:
    tensor: (`numpy.ndarray`) the tensor value object to be summarized.

  Returns:
    The summary text as a `RichTextLines` object. If the type of `tensor` is not
    numeric or Boolean, a single-line `RichTextLines` object containing a
    warning message will reflect that.
  """

    def _counts_summary(counts, skip_zeros=True, total_count=None):
        """Format values as a two-row table."""
        if skip_zeros:
            counts = [(count_key, count_val) for count_key, count_val in counts if count_val]
        max_common_len = 0
        for count_key, count_val in counts:
            count_val_str = str(count_val)
            common_len = max(len(count_key) + 1, len(count_val_str) + 1)
            max_common_len = max(common_len, max_common_len)
        key_line = debugger_cli_common.RichLine('|')
        val_line = debugger_cli_common.RichLine('|')
        for count_key, count_val in counts:
            count_val_str = str(count_val)
            key_line += _pad_string_to_length(count_key, max_common_len)
            val_line += _pad_string_to_length(count_val_str, max_common_len)
        key_line += ' |'
        val_line += ' |'
        if total_count is not None:
            total_key_str = 'total'
            total_val_str = str(total_count)
            max_common_len = max(len(total_key_str) + 1, len(total_val_str))
            total_key_str = _pad_string_to_length(total_key_str, max_common_len)
            total_val_str = _pad_string_to_length(total_val_str, max_common_len)
            key_line += total_key_str + ' |'
            val_line += total_val_str + ' |'
        return debugger_cli_common.rich_text_lines_from_rich_line_list([key_line, val_line])
    if not isinstance(tensor, np.ndarray) or not np.size(tensor):
        return debugger_cli_common.RichTextLines(['No numeric summary available due to empty tensor.'])
    elif np.issubdtype(tensor.dtype, np.floating) or np.issubdtype(tensor.dtype, np.complexfloating) or np.issubdtype(tensor.dtype, np.integer):
        counts = [('nan', np.sum(np.isnan(tensor))), ('-inf', np.sum(np.isneginf(tensor))), ('-', np.sum(np.logical_and(tensor < 0.0, np.logical_not(np.isneginf(tensor))))), ('0', np.sum(tensor == 0.0)), ('+', np.sum(np.logical_and(tensor > 0.0, np.logical_not(np.isposinf(tensor))))), ('+inf', np.sum(np.isposinf(tensor)))]
        output = _counts_summary(counts, total_count=np.size(tensor))
        valid_array = tensor[np.logical_not(np.logical_or(np.isinf(tensor), np.isnan(tensor)))]
        if np.size(valid_array):
            stats = [('min', np.min(valid_array)), ('max', np.max(valid_array)), ('mean', np.mean(valid_array)), ('std', np.std(valid_array))]
            output.extend(_counts_summary(stats, skip_zeros=False))
        return output
    elif tensor.dtype == np.bool_:
        counts = [('False', np.sum(tensor == 0)), ('True', np.sum(tensor > 0))]
        return _counts_summary(counts, total_count=np.size(tensor))
    else:
        return debugger_cli_common.RichTextLines(['No numeric summary available due to tensor dtype: %s.' % tensor.dtype])