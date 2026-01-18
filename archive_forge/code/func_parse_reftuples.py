from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def parse_reftuples(lh_container: Union['Repo', 'RefsContainer'], rh_container: Union['Repo', 'RefsContainer'], refspecs: Union[bytes, List[bytes]], force: bool=False):
    """Parse a list of reftuple specs to a list of reftuples.

    Args:
      lh_container: A RefsContainer object
      rh_container: A RefsContainer object
      refspecs: A list of refspecs or a string
      force: Force overwriting for all reftuples
    Returns: A list of refs
    Raises:
      KeyError: If one of the refs can not be found
    """
    if not isinstance(refspecs, list):
        refspecs = [refspecs]
    ret = []
    for refspec in refspecs:
        ret.append(parse_reftuple(lh_container, rh_container, refspec, force=force))
    return ret