from __future__ import annotations
import sys
import eventlet
def slurp_properties(source, destination, ignore=[], srckeys=None):
    """Copy properties from *source* (assumed to be a module) to
    *destination* (assumed to be a dict).

    *ignore* lists properties that should not be thusly copied.
    *srckeys* is a list of keys to copy, if the source's __all__ is
    untrustworthy.
    """
    if srckeys is None:
        srckeys = source.__all__
    destination.update({name: getattr(source, name) for name in srckeys if not (name.startswith('__') or name in ignore)})