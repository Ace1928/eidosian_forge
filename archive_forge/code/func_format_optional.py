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
def format_optional(opt, parser):
    get_help = parser._get_formatter()._expand_help
    return ('{nargs}{options}"[{help}]"' if isinstance(opt, FLAG_OPTION) else '{nargs}{options}"[{help}]:{dest}:{pattern}"').format(nargs='"(- : *)"' if is_opt_end(opt) else '"*"' if is_opt_multiline(opt) else '', options='{{{}}}'.format(','.join(opt.option_strings)) if len(opt.option_strings) > 1 else '"{}"'.format(''.join(opt.option_strings)), help=escape_zsh(get_help(opt) if opt.help else ''), dest=opt.dest, pattern=complete2pattern(opt.complete, 'zsh', choice_type2fn) if hasattr(opt, 'complete') else (choice_type2fn[opt.choices[0].type] if isinstance(opt.choices[0], Choice) else '({})'.format(' '.join(map(str, opt.choices)))) if opt.choices else '').replace('""', '')