import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union
from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size
from .utils.insecure_hashlib import sha256
def multi_commit_parse_pr_description(description: str) -> Set[str]:
    return {match[1] for match in STEP_ID_REGEX.findall(description)}