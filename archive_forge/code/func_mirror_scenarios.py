import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def mirror_scenarios(base_scenarios):
    """Return a list of mirrored scenarios.

    Each scenario in base_scenarios is duplicated switching the roles of 'this'
    and 'other'
    """
    scenarios = []
    for common, (lname, ldict), (rname, rdict) in base_scenarios:
        a = tests.multiply_scenarios([(lname, dict(_this=ldict))], [(rname, dict(_other=rdict))])
        b = tests.multiply_scenarios([(rname, dict(_this=rdict))], [(lname, dict(_other=ldict))])
        for name, d in a + b:
            d.update(common)
        scenarios.extend(a + b)
    return scenarios