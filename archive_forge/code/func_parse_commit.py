from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def parse_commit(repo: 'Repo', committish: Union[str, bytes]) -> 'Commit':
    """Parse a string referring to a single commit.

    Args:
      repo: A` Repo` object
      committish: A string referring to a single commit.
    Returns: A Commit object
    Raises:
      KeyError: When the reference commits can not be found
      ValueError: If the range can not be parsed
    """
    committish = to_bytes(committish)
    try:
        return repo[committish]
    except KeyError:
        pass
    try:
        return repo[parse_ref(repo, committish)]
    except KeyError:
        pass
    if len(committish) >= 4 and len(committish) < 40:
        try:
            int(committish, 16)
        except ValueError:
            pass
        else:
            try:
                return scan_for_short_id(repo.object_store, committish)
            except KeyError:
                pass
    raise KeyError(committish)