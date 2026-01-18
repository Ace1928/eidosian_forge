from yaql.language import exceptions
import yaql.tests
def test_join_pythonic(self):
    self.assertEqual('some-text', self.eval("'-'.join([some, text])"))