import re
from difflib import SequenceMatcher, unified_diff
from pyomo.repn.tests.diffutils import compare_floats, load_baseline
def load_and_compare_lp_baseline(baseline, testfile, version='lp'):
    return lp_diff(*load_lp_baseline(baseline, testfile, version))