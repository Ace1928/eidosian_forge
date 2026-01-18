from argparse import (
from gettext import gettext
from typing import Dict, List, Set, Tuple
def consume_optional(start_index):
    option_tuple = option_string_indices[start_index]
    action, option_string, explicit_arg = option_tuple
    match_argument = self._match_argument
    action_tuples: List[Tuple[Action, List[str], str]] = []
    while True:
        if action is None:
            extras.append(arg_strings[start_index])
            return start_index + 1
        if explicit_arg is not None:
            arg_count = match_argument(action, 'A')
            chars = self.prefix_chars
            if arg_count == 0 and option_string[1] not in chars:
                action_tuples.append((action, [], option_string))
                char = option_string[0]
                option_string = char + explicit_arg[0]
                new_explicit_arg = explicit_arg[1:] or None
                optionals_map = self._option_string_actions
                if option_string in optionals_map:
                    action = optionals_map[option_string]
                    explicit_arg = new_explicit_arg
                else:
                    msg = gettext('ignored explicit argument %r')
                    raise ArgumentError(action, msg % explicit_arg)
            elif arg_count == 1:
                stop = start_index + 1
                args = [explicit_arg]
                action_tuples.append((action, args, option_string))
                break
            else:
                msg = gettext('ignored explicit argument %r')
                raise ArgumentError(action, msg % explicit_arg)
        else:
            start = start_index + 1
            selected_patterns = arg_strings_pattern[start:]
            self.active_actions = [action]
            _num_consumed_args[action] = 0
            arg_count = match_argument(action, selected_patterns)
            stop = start + arg_count
            args = arg_strings[start:stop]
            _num_consumed_args[action] = len(args)
            if not action_is_open(action):
                self.active_actions.remove(action)
            action_tuples.append((action, args, option_string))
            break
    assert action_tuples
    for action, args, option_string in action_tuples:
        take_action(action, args, option_string)
    return stop