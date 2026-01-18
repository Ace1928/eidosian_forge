from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def filtered_output_bytes(chunks, filters, context=None):
    """Convert byte chunks from internal to external format.

    Args:
      chunks: an iterator containing the original content
      filters: the stack of filters to apply
      context: a ContentFilterContext object passed to
        each filter

    Returns: an iterator containing the content to output
    """
    if filters:
        for filter in reversed(filters):
            if filter.writer is not None:
                chunks = filter.writer(chunks, context)
    return chunks