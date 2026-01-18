from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
def dump_transitions(self, state, file):
    chars_leading_to_state = {}
    special_to_state = {}
    for c, s in state.items():
        if len(c) == 1:
            chars = chars_leading_to_state.get(id(s), None)
            if chars is None:
                chars = []
                chars_leading_to_state[id(s)] = chars
            chars.append(c)
        elif len(c) <= 4:
            special_to_state[c] = s
    ranges_to_state = {}
    for state in self.states:
        char_list = chars_leading_to_state.get(id(state), None)
        if char_list:
            ranges = self.chars_to_ranges(char_list)
            ranges_to_state[ranges] = state
    for ranges in sorted(ranges_to_state):
        key = self.ranges_to_string(ranges)
        state = ranges_to_state[ranges]
        file.write('      %s --> State %d\n' % (key, state['number']))
    for key in ('bol', 'eol', 'eof', 'else'):
        state = special_to_state.get(key, None)
        if state:
            file.write('      %s --> State %d\n' % (key, state['number']))