import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union
from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size
from .utils.insecure_hashlib import sha256
def multi_commit_generate_comment(commit_message: str, commit_description: Optional[str], strategy: MultiCommitStrategy) -> str:
    return MULTI_COMMIT_PR_DESCRIPTION_TEMPLATE.format(commit_message=commit_message, commit_description=commit_description or '', multi_commit_id=strategy.id, multi_commit_strategy='\n'.join((str(commit) for commit in strategy.deletion_commits + strategy.addition_commits)))