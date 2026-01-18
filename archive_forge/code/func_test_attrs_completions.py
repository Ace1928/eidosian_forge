from .common import TestCase
def test_attrs_completions(self):
    attrs = self.f.attrs
    attrs['b'] = 1
    attrs['a'] = 2
    self.assertEqual(attrs._ipython_key_completions_(), ['a', 'b'])
    attrs['c'] = 3
    self.assertEqual(attrs._ipython_key_completions_(), ['a', 'b', 'c'])