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
def get_public_subcommands(sub):
    """Get all the publicly-visible subcommands for a given subparser."""
    public_parsers = {id(sub.choices[i.dest]) for i in sub._get_subactions()}
    return {k for k, v in sub.choices.items() if id(v) in public_parsers}