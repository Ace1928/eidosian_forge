from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def is_null(revision_id: RevisionID) -> bool:
    if revision_id is None:
        raise ValueError('NULL_REVISION should be used for the null revision instead of None.')
    return revision_id == NULL_REVISION