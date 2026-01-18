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
def escape_zsh(string):
    return re.sub('([^\\w\\s.,()-])', '\\\\\\1', str(string))