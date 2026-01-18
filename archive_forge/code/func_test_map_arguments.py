import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
def test_map_arguments(self):
    objs = []

    def mapped(ext, *args, **kwds):
        objs.append((ext, args, kwds))
    em = extension.ExtensionManager('stevedore.test.extension', invoke_on_load=True)
    em.map(mapped, 1, 2, a='A', b='B')
    self.assertEqual(len(objs), 2)
    names = sorted([o[0].name for o in objs])
    self.assertEqual(names, WORKING_NAMES)
    for o in objs:
        self.assertEqual(o[1], (1, 2))
        self.assertEqual(o[2], {'a': 'A', 'b': 'B'})