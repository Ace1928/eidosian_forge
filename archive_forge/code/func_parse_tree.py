from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def parse_tree(repo: 'Repo', treeish: Union[bytes, str]) -> 'Tree':
    """Parse a string referring to a tree.

    Args:
      repo: A `Repo` object
      treeish: A string referring to a tree
    Returns: A git object
    Raises:
      KeyError: If the object can not be found
    """
    treeish = to_bytes(treeish)
    try:
        treeish = parse_ref(repo, treeish)
    except KeyError:
        pass
    o = repo[treeish]
    if o.type_name == b'commit':
        return repo[o.tree]
    return o