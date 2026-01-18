import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_indexation_blacklist(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj, blacklist=[lambda t: t.startswith('_')])
    self.assertEqual('key', self.eval('$[key]', obj))
    self.assertRaises(KeyError, self.eval, '$[_key]', obj)