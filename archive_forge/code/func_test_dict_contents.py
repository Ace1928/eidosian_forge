import dill
from dill import objects
from dill import load_types
def test_dict_contents():
    c = type.__dict__
    for i, j in c.items():
        ok = dill.pickles(j)
        if verbose:
            print('%s: %s, %s' % (ok, type(j), j))
        assert ok
    if verbose:
        print('')