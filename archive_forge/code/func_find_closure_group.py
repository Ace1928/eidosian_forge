import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def find_closure_group(input_string, start, group):
    """Generalization for finding a balancing closure group

         if group = ["(", ")"], then finds the first balanced parentheses.
         if group = ["{", "}"], then finds the first balanced bracket.

    Given an input string, a starting position in the input string, and the group type,
    find_closure_group returns the positions of group[0] and group[1] as a tuple.

    Example:
        >>> find_closure_group("(hi)", 0, ["(", ")"])
        (0, 3)
    """
    inside_parenthesis = False
    parens = 0
    pos = start
    p_start, p_end = (-1, -1)
    while pos < len(input_string):
        if input_string[pos] == group[0]:
            if inside_parenthesis is False:
                inside_parenthesis = True
                parens = 1
                p_start = pos
            else:
                parens += 1
        elif input_string[pos] == group[1] and inside_parenthesis:
            parens -= 1
            if parens == 0:
                p_end = pos
                return (p_start, p_end)
        pos += 1
    return (None, None)