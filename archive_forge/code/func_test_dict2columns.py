import weakref
from unittest import mock
from cliff import show
from cliff.tests import base
def test_dict2columns(self):
    app = mock.Mock()
    test_show = ExerciseShowOne(app, [])
    d = {'a': 'A', 'b': 'B', 'c': 'C'}
    expected = [('a', 'b', 'c'), ('A', 'B', 'C')]
    actual = list(test_show.dict2columns(d))
    self.assertEqual(expected, actual)