from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def is_reserved_id(revision_id: RevisionID) -> bool:
    """Determine whether a revision id is reserved

    Returns:
      True if the revision is reserved, False otherwise
    """
    return isinstance(revision_id, bytes) and revision_id.endswith(b':')