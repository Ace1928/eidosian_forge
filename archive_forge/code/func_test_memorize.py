from yaql.language import exceptions
import yaql.tests
def test_memorize(self):
    generator_func = lambda: (i for i in range(3))
    self.assertRaises(TypeError, self.eval, '$.len() + $.sum()', data=generator_func())
    self.assertEqual(6, self.eval('let($.memorize()) -> $.len() + $.sum()', data=generator_func()))