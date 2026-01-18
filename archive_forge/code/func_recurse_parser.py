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
def recurse_parser(cparser, positional_idx, requirements=None):
    log_prefix = '| ' * positional_idx
    log.debug('%sParser @ %d', log_prefix, positional_idx)
    if requirements:
        log.debug('%s- Requires: %s', log_prefix, ' '.join(requirements))
    else:
        requirements = []
    for optional in cparser._get_optional_actions():
        log.debug('%s| Optional: %s', log_prefix, optional.dest)
        if optional.help != SUPPRESS:
            for optional_str in optional.option_strings:
                log.debug('%s| | %s', log_prefix, optional_str)
                if optional_str.startswith('--'):
                    optionals_double.add(optional_str[2:])
                elif optional_str.startswith('-'):
                    optionals_single.add(optional_str[1:])
                specials.extend(get_specials(optional, 'n', optional_str))
    for positional in cparser._get_positional_actions():
        if positional.help != SUPPRESS:
            positional_idx += 1
            log.debug('%s| Positional #%d: %s', log_prefix, positional_idx, positional.dest)
            index_choices[positional_idx][tuple(requirements)] = positional
            if not requirements and isinstance(positional.choices, dict):
                for subcmd, subparser in positional.choices.items():
                    log.debug('%s| | SubParser: %s', log_prefix, subcmd)
                    recurse_parser(subparser, positional_idx, requirements + [subcmd])