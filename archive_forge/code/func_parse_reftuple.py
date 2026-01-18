from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def parse_reftuple(lh_container: Union['Repo', 'RefsContainer'], rh_container: Union['Repo', 'RefsContainer'], refspec: Union[str, bytes], force: bool=False) -> Tuple[Optional['Ref'], Optional['Ref'], bool]:
    """Parse a reftuple spec.

    Args:
      lh_container: A RefsContainer object
      rh_container: A RefsContainer object
      refspec: A string
    Returns: A tuple with left and right ref
    Raises:
      KeyError: If one of the refs can not be found
    """
    refspec = to_bytes(refspec)
    if refspec.startswith(b'+'):
        force = True
        refspec = refspec[1:]
    lh: Optional[bytes]
    rh: Optional[bytes]
    if b':' in refspec:
        lh, rh = refspec.split(b':')
    else:
        lh = rh = refspec
    if lh == b'':
        lh = None
    else:
        lh = parse_ref(lh_container, lh)
    if rh == b'':
        rh = None
    else:
        try:
            rh = parse_ref(rh_container, rh)
        except KeyError:
            if b'/' not in rh:
                rh = b'refs/heads/' + rh
    return (lh, rh, force)