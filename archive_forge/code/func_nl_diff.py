import itertools
import re
from difflib import SequenceMatcher, unified_diff
from pyomo.repn.tests.diffutils import compare_floats, load_baseline
import pyomo.repn.plugins.nl_writer as nl_writer
def nl_diff(base, test, baseline='baseline', testfile='testfile'):
    if test == base:
        return ([], [])
    test = _preprocess_data(test)
    base = _preprocess_data(base)
    if test == base:
        return ([], [])
    test_nlines = list((x for x in enumerate(test) if x[1] and x[1][0] == 'n'))
    base_nlines = list((x for x in enumerate(base) if x[1] and x[1][0] == 'n'))
    if len(test_nlines) == len(base_nlines):
        for t_line, b_line in zip(test_nlines, base_nlines):
            if compare_floats(t_line[1][1:], b_line[1][1:]):
                test[t_line[0]] = base[b_line[0]]
    for group in SequenceMatcher(None, base, test).get_grouped_opcodes(3):
        for tag, i1, i2, j1, j2 in group:
            if tag != 'replace':
                continue
            _update_subsets((range(i1, i2), range(j1, j2)), base, test)
    if test == base:
        return ([], [])
    print(''.join(unified_diff([_ + '\n' for _ in base], [_ + '\n' for _ in test], fromfile=baseline, tofile=testfile)))
    return (base, test)