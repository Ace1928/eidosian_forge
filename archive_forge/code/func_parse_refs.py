from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def parse_refs(container, refspecs):
    """Parse a list of refspecs to a list of refs.

    Args:
      container: A RefsContainer object
      refspecs: A list of refspecs or a string
    Returns: A list of refs
    Raises:
      KeyError: If one of the refs can not be found
    """
    if not isinstance(refspecs, list):
        refspecs = [refspecs]
    ret = []
    for refspec in refspecs:
        ret.append(parse_ref(container, refspec))
    return ret