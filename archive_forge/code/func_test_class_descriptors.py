import dill
from dill import objects
from dill import load_types
def test_class_descriptors():
    d = _d.__dict__
    for i in d.values():
        ok = dill.pickles(i)
        if verbose:
            print('%s: %s, %s' % (ok, type(i), i))
        assert ok
    if verbose:
        print('')
    od = _newclass.__dict__
    for i in od.values():
        ok = dill.pickles(i)
        if verbose:
            print('%s: %s, %s' % (ok, type(i), i))
        assert ok
    if verbose:
        print('')