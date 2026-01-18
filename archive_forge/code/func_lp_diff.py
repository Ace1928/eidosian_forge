import re
from difflib import SequenceMatcher, unified_diff
from pyomo.repn.tests.diffutils import compare_floats, load_baseline
def lp_diff(base, test, baseline='baseline', testfile='testfile'):
    if test == base:
        return ([], [])
    test = list(_preprocess_data(test))
    base = list(_preprocess_data(base))
    if test == base:
        return ([], [])
    for group in SequenceMatcher(None, base, test).get_grouped_opcodes(3):
        for tag, i1, i2, j1, j2 in group:
            if tag != 'replace':
                continue
            _update_subsets((range(i1, i2), range(j1, j2)), base, test)
    if test == base:
        return ([], [])
    print(''.join(unified_diff([_ + '\n' for _ in base], [_ + '\n' for _ in test], fromfile=baseline, tofile=testfile)))
    return (base, test)