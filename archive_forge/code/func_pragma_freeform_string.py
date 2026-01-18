import json
import pkgutil
import operator
from typing import List
from deprecated import deprecated
from deprecated.sphinx import versionadded
from lark import Lark, Transformer, v_args
import numpy as np
from pyquil.quilbase import (
from pyquil.quiltwaveforms import _wf_from_dict
from pyquil.quilatom import (
from pyquil.gates import (
@v_args(inline=True)
def pragma_freeform_string(self, name, *pragma_names_and_string):
    if len(pragma_names_and_string) == 1:
        freeform_string = pragma_names_and_string[0]
        args = ()
    else:
        *args_identifiers, freeform_string = pragma_names_and_string
        args = list(map(str, args_identifiers))
    freeform_string = freeform_string[1:-1]
    p = Pragma(str(name), args=args, freeform_string=freeform_string)
    return p