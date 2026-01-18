from argparse import (
from gettext import gettext
from typing import Dict, List, Set, Tuple
def consume_positionals(start_index):
    match_partial = self._match_arguments_partial
    selected_pattern = arg_strings_pattern[start_index:]
    arg_counts = match_partial(positionals, selected_pattern)
    for action, arg_count in zip(positionals, arg_counts):
        self.active_actions.append(action)
    for action, arg_count in zip(positionals, arg_counts):
        args = arg_strings[start_index:start_index + arg_count]
        start_index += arg_count
        _num_consumed_args[action] = len(args)
        take_action(action, args)
    positionals[:] = positionals[len(arg_counts):]
    return start_index