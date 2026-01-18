import unittest
from IPython.utils import wildcard
def test_nocase_showall(self):
    ns = root.__dict__
    tests = [('a*', ['abbot', 'abel', 'ABEL', 'active', 'arna']), ('?b*.?o*', ['abbot.koppel', 'abbot.loop', 'abel.koppel', 'abel.loop', 'ABEL.koppel', 'ABEL.loop']), ('_a*', ['_apan', '_APAN']), ('_*anka', ['__anka', '__ANKA']), ('_*a*', ['__anka', '__ANKA', '_apan', '_APAN'])]
    for pat, res in tests:
        res.sort()
        a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=True, show_all=True).keys())
        a.sort()
        self.assertEqual(a, res)