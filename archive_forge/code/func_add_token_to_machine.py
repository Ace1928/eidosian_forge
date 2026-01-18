from __future__ import absolute_import
from . import Actions
from . import DFA
from . import Errors
from . import Machines
from . import Regexps
def add_token_to_machine(self, machine, initial_state, token_spec, token_number):
    try:
        re, action_spec = self.parse_token_definition(token_spec)
        if isinstance(action_spec, Actions.Action):
            action = action_spec
        else:
            try:
                action_spec.__call__
            except AttributeError:
                action = Actions.Return(action_spec)
            else:
                action = Actions.Call(action_spec)
        final_state = machine.new_state()
        re.build_machine(machine, initial_state, final_state, match_bol=1, nocase=0)
        final_state.set_action(action, priority=-token_number)
    except Errors.PlexError as e:
        raise e.__class__('Token number %d: %s' % (token_number, e))