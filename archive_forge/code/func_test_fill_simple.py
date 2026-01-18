from .. import tests, utextwrap
def test_fill_simple(self):
    self.assertEqual('{}\n{}'.format(_str_D[:2], _str_D[2:]), utextwrap.fill(_str_D, 4))