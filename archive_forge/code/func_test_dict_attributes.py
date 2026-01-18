import unittest
from IPython.utils import wildcard
def test_dict_attributes(self):
    """Dictionaries should be indexed by attributes, not by keys. This was
        causing Github issue 129."""
    ns = {'az': {'king': 55}, 'pq': {1: 0}}
    tests = [('a*', ['az']), ('az.k*', ['az.keys']), ('pq.k*', ['pq.keys'])]
    for pat, res in tests:
        res.sort()
        a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=False, show_all=True).keys())
        self.assertEqual(a, res)