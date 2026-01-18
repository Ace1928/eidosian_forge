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
def complete2pattern(opt_complete, shell: str, choice_type2fn) -> str:
    return opt_complete.get(shell, '') if isinstance(opt_complete, dict) else choice_type2fn[opt_complete]