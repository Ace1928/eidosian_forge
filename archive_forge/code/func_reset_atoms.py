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
def reset_atoms(self, atoms, state=st.PENDING, intention=st.EXECUTE):
    """Resets all the provided atoms to the given state and intention."""
    tweaked = []
    for atom in atoms:
        if state or intention:
            tweaked.append((atom, state, intention))
        if state:
            change_state_handler = self._fetch_atom_metadata_entry(atom.name, 'change_state_handler')
            change_state_handler(atom, state)
        if intention:
            self.storage.set_atom_intention(atom.name, intention)
    return tweaked