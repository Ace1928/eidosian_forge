from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
from monty.functools import lazy_property
from monty.json import MSONable
from pymatgen.core.libxcfunc import LibxcFunc
@classmethod
def from_abinit_ixc(cls, ixc: int) -> Self | None:
    """Build the object from Abinit ixc (integer)."""
    if ixc == 0:
        return None
    if ixc > 0:
        return cls(**cls.abinitixc_to_libxc[ixc])
    ixc = abs(ixc)
    first = ixc // 1000
    last = ixc - first * 1000
    x, c = (LibxcFunc(int(first)), LibxcFunc(int(last)))
    if not x.is_x_kind:
        x, c = (c, x)
    assert x.is_x_kind
    assert c.is_c_kind
    return cls(x=x, c=c)