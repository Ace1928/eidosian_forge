import collections
from enum import Enum
from typing import Any, Callable, Dict, List
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import identity, istype
def can_reconstruct(self, tx):
    """If it is possible to reconstruct the Python object this
        VariableTracker represents."""
    assert tx is tx.output.root_tx, 'Only root tx can reconstruct'
    try:
        from ..codegen import PyCodegen
        cg = PyCodegen(tx)
        self.reconstruct(cg)
        return True
    except NotImplementedError:
        return False