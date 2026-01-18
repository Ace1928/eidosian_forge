import operator
import weakref
from taskflow.engines.action_engine import compiler as co
from taskflow.engines.action_engine import deciders
from taskflow.engines.action_engine import traversal
from taskflow import logging
from taskflow import states as st
from taskflow.utils import iter_utils
def ready_checker(succ_connected_it):
    for succ in succ_connected_it:
        succ_atom, (succ_atom_state, _succ_atom_intention) = succ
        if succ_atom_state not in (st.PENDING, st.REVERTED, st.IGNORE):
            LOG.trace("Unable to begin to revert since successor atom '%s' is in state %s", succ_atom, succ_atom_state)
            return False
    LOG.trace("Able to let '%s' revert", atom)
    return True