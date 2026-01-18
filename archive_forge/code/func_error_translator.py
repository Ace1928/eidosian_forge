from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export
def error_translator(e):
    """Translate the tensor_slice_reader.cc errors."""
    error_message = str(e)
    if 'not found in checkpoint' in error_message or 'Failed to find any matching files for' in error_message:
        raise errors_impl.NotFoundError(None, None, error_message)
    elif 'Sliced checkpoints are not supported' in error_message or 'Data type not supported' in error_message:
        raise errors_impl.UnimplementedError(None, None, error_message)
    elif 'Failed to get matching files on' in error_message:
        raise errors_impl.InvalidArgumentError(None, None, error_message)
    elif 'Unable to open table file' in error_message:
        raise errors_impl.DataLossError(None, None, error_message)
    elif 'Failed to find the saved tensor slices' in error_message or 'not convertible to numpy dtype' in error_message:
        raise errors_impl.InternalError(None, None, error_message)
    else:
        raise errors_impl.OpError(None, None, error_message, errors_impl.UNKNOWN)