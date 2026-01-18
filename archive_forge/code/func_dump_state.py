from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
def dump_state(self, state, file):
    file.write('   State %d:\n' % state['number'])
    self.dump_transitions(state, file)
    action = state['action']
    if action is not None:
        file.write('      %s\n' % action)