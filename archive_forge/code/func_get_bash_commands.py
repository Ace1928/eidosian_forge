import logging
import re
from argparse import (
from collections import defaultdict
from functools import total_ordering
from itertools import starmap
from string import Template
from typing import Any, Dict, List
from typing import Optional as Opt
from typing import Union
def get_bash_commands(root_parser, root_prefix, choice_functions=None):
    """
    Recursive subcommand parser traversal, returning lists of information on
    commands (formatted for output to the completions script).
    printing bash helper syntax.

    Returns:
      subparsers  : list of subparsers for each parser
      option_strings  : list of options strings for each parser
      compgens  : list of shtab `.complete` functions corresponding to actions
      choices  : list of choices corresponding to actions
      nargs  : list of number of args allowed for each action (if not 0 or 1)
    """
    choice_type2fn = {k: v['bash'] for k, v in CHOICE_FUNCTIONS.items()}
    if choice_functions:
        choice_type2fn.update(choice_functions)

    def get_option_strings(parser):
        """Flattened list of all `parser`'s option strings."""
        return sum((opt.option_strings for opt in parser._get_optional_actions() if opt.help != SUPPRESS), [])

    def recurse(parser, prefix):
        """recurse through subparsers, appending to the return lists"""
        subparsers = []
        option_strings = []
        compgens = []
        choices = []
        nargs = []
        sub_subparsers = []
        sub_option_strings = []
        sub_compgens = []
        sub_choices = []
        sub_nargs = []
        discovered_subparsers = []
        for i, positional in enumerate(parser._get_positional_actions()):
            if positional.help == SUPPRESS:
                continue
            if hasattr(positional, 'complete'):
                comp_pattern = complete2pattern(positional.complete, 'bash', choice_type2fn)
                compgens.append(f'{prefix}_pos_{i}_COMPGEN={comp_pattern}')
            if positional.choices:
                log.debug(f'choices:{prefix}:{sorted(positional.choices)}')
                this_positional_choices = []
                for choice in positional.choices:
                    if isinstance(choice, Choice):
                        log.debug(f'Choice.{choice.type}:{prefix}:{positional.dest}')
                        compgens.append(f'{prefix}_pos_{i}_COMPGEN={choice_type2fn[choice.type]}')
                    elif isinstance(positional.choices, dict):
                        log.debug('subcommand:%s', choice)
                        public_cmds = get_public_subcommands(positional)
                        if choice in public_cmds:
                            discovered_subparsers.append(str(choice))
                            this_positional_choices.append(str(choice))
                            new_subparsers, new_option_strings, new_compgens, new_choices, new_nargs = recurse(positional.choices[choice], f'{prefix}_{wordify(choice)}')
                            sub_subparsers.extend(new_subparsers)
                            sub_option_strings.extend(new_option_strings)
                            sub_compgens.extend(new_compgens)
                            sub_choices.extend(new_choices)
                            sub_nargs.extend(new_nargs)
                        else:
                            log.debug('skip:subcommand:%s', choice)
                    else:
                        this_positional_choices.append(str(choice))
                if this_positional_choices:
                    choices_str = "' '".join(this_positional_choices)
                    choices.append(f"{prefix}_pos_{i}_choices=('{choices_str}')")
            if positional.nargs not in (None, '1', '?'):
                nargs.append(f'{prefix}_pos_{i}_nargs={positional.nargs}')
        if discovered_subparsers:
            subparsers_str = "' '".join(discovered_subparsers)
            subparsers.append(f"{prefix}_subparsers=('{subparsers_str}')")
            log.debug(f'subcommands:{prefix}:{discovered_subparsers}')
        options_strings_str = "' '".join(get_option_strings(parser))
        option_strings.append(f"{prefix}_option_strings=('{options_strings_str}')")
        for optional in parser._get_optional_actions():
            if optional == SUPPRESS:
                continue
            for option_string in optional.option_strings:
                if hasattr(optional, 'complete'):
                    comp_pattern_str = complete2pattern(optional.complete, 'bash', choice_type2fn)
                    compgens.append(f'{prefix}_{wordify(option_string)}_COMPGEN={comp_pattern_str}')
                if optional.choices:
                    this_optional_choices = []
                    for choice in optional.choices:
                        if isinstance(choice, Choice):
                            log.debug(f'Choice.{choice.type}:{prefix}:{optional.dest}')
                            func_str = choice_type2fn[choice.type]
                            compgens.append(f'{prefix}_{wordify(option_string)}_COMPGEN={func_str}')
                        else:
                            this_optional_choices.append(str(choice))
                    if this_optional_choices:
                        this_choices_str = "' '".join(this_optional_choices)
                        choices.append(f"{prefix}_{wordify(option_string)}_choices=('{this_choices_str}')")
                if optional.nargs is not None and optional.nargs != 1:
                    nargs.append(f'{prefix}_{wordify(option_string)}_nargs={optional.nargs}')
        subparsers.extend(sub_subparsers)
        option_strings.extend(sub_option_strings)
        compgens.extend(sub_compgens)
        choices.extend(sub_choices)
        nargs.extend(sub_nargs)
        return (subparsers, option_strings, compgens, choices, nargs)
    return recurse(root_parser, root_prefix)