import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union
from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size
from .utils.insecure_hashlib import sha256
@experimental
def plan_multi_commits(operations: Iterable[Union[CommitOperationAdd, CommitOperationDelete]], max_operations_per_commit: int=50, max_upload_size_per_commit: int=2 * 1024 * 1024 * 1024) -> Tuple[List[List[CommitOperationAdd]], List[List[CommitOperationDelete]]]:
    """Split a list of operations in a list of commits to perform.

    Implementation follows a sub-optimal (yet simple) algorithm:
    1. Delete operations are grouped together by commits of maximum `max_operations_per_commits` operations.
    2. All additions exceeding `max_upload_size_per_commit` are committed 1 by 1.
    3. All remaining additions are grouped together and split each time the `max_operations_per_commit` or the
       `max_upload_size_per_commit` limit is reached.

    We do not try to optimize the splitting to get the lowest number of commits as this is a NP-hard problem (see
    [bin packing problem](https://en.wikipedia.org/wiki/Bin_packing_problem)). For our use case, it is not problematic
    to use a sub-optimal solution so we favored an easy-to-explain implementation.

    Args:
        operations (`List` of [`~hf_api.CommitOperation`]):
            The list of operations to split into commits.
        max_operations_per_commit (`int`):
            Maximum number of operations in a single commit. Defaults to 50.
        max_upload_size_per_commit (`int`):
            Maximum size to upload (in bytes) in a single commit. Defaults to 2GB. Files bigger than this limit are
            uploaded, 1 per commit.

    Returns:
        `Tuple[List[List[CommitOperationAdd]], List[List[CommitOperationDelete]]]`: a tuple. First item is a list of
        lists of [`CommitOperationAdd`] representing the addition commits to push. The second item is a list of lists
        of [`CommitOperationDelete`] representing the deletion commits.

    <Tip warning={true}>

    `plan_multi_commits` is experimental. Its API and behavior is subject to change in the future without prior notice.

    </Tip>

    Example:
    ```python
    >>> from huggingface_hub import HfApi, plan_multi_commits
    >>> addition_commits, deletion_commits = plan_multi_commits(
    ...     operations=[
    ...          CommitOperationAdd(...),
    ...          CommitOperationAdd(...),
    ...          CommitOperationDelete(...),
    ...          CommitOperationDelete(...),
    ...          CommitOperationAdd(...),
    ...     ],
    ... )
    >>> HfApi().create_commits_on_pr(
    ...     repo_id="my-cool-model",
    ...     addition_commits=addition_commits,
    ...     deletion_commits=deletion_commits,
    ...     (...)
    ...     verbose=True,
    ... )
    ```

    <Tip warning={true}>

    The initial order of the operations is not guaranteed! All deletions will be performed before additions. If you are
    not updating multiple times the same file, you are fine.

    </Tip>
    """
    addition_commits: List[List[CommitOperationAdd]] = []
    deletion_commits: List[List[CommitOperationDelete]] = []
    additions: List[CommitOperationAdd] = []
    additions_size = 0
    deletions: List[CommitOperationDelete] = []
    for op in operations:
        if isinstance(op, CommitOperationDelete):
            deletions.append(op)
            if len(deletions) >= max_operations_per_commit:
                deletion_commits.append(deletions)
                deletions = []
        elif op.upload_info.size >= max_upload_size_per_commit:
            addition_commits.append([op])
        elif additions_size + op.upload_info.size < max_upload_size_per_commit:
            additions.append(op)
            additions_size += op.upload_info.size
        else:
            addition_commits.append(additions)
            additions = [op]
            additions_size = op.upload_info.size
        if len(additions) >= max_operations_per_commit:
            addition_commits.append(additions)
            additions = []
            additions_size = 0
    if len(additions) > 0:
        addition_commits.append(additions)
    if len(deletions) > 0:
        deletion_commits.append(deletions)
    return (addition_commits, deletion_commits)