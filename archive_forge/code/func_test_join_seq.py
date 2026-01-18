from yaql.language import exceptions
import yaql.tests
def test_join_seq(self):
    self.assertEqual('text-1-null-true', self.eval("[text, 1, null, true].select(str($)).join('-')"))