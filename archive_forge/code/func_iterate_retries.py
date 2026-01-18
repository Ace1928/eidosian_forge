import collections
import functools
from futurist import waiters
from taskflow import deciders as de
from taskflow.engines.action_engine.actions import retry as ra
from taskflow.engines.action_engine.actions import task as ta
from taskflow.engines.action_engine import builder as bu
from taskflow.engines.action_engine import compiler as com
from taskflow.engines.action_engine import completer as co
from taskflow.engines.action_engine import scheduler as sched
from taskflow.engines.action_engine import scopes as sc
from taskflow.engines.action_engine import selector as se
from taskflow.engines.action_engine import traversal as tr
from taskflow import exceptions as exc
from taskflow import logging
from taskflow import states as st
from taskflow.utils import misc
from taskflow.flow import (LINK_DECIDER, LINK_DECIDER_DEPTH)  # noqa
def iterate_retries(self, state=None):
    """Iterates retry atoms that match the provided state.

        If no state is provided it will yield back all retry atoms.
        """
    if state:
        atoms = list(self.iterate_nodes((com.RETRY,)))
        atom_states = self._storage.get_atoms_states((atom.name for atom in atoms))
        for atom in atoms:
            atom_state, _atom_intention = atom_states[atom.name]
            if atom_state == state:
                yield atom
    else:
        for atom in self.iterate_nodes((com.RETRY,)):
            yield atom