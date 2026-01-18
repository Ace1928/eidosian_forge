from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def internal_size_sha_file_byname(name, filters):
    """Get size and sha of internal content given external content.

    Args:
      name: path to file
      filters: the stack of filters to apply
    """
    with open(name, 'rb', 65000) as f:
        if filters:
            f, size = filtered_input_file(f, filters)
        return osutils.size_sha_file(f)