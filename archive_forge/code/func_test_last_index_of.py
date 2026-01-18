from yaql.language import exceptions
import yaql.tests
def test_last_index_of(self):
    self.assertEqual(3, self.eval('[1, 2, 3, 2, 1].lastIndexOf(2)'))
    self.assertEqual(-1, self.eval('[1, 2, 3, 2, 1].lastIndexOf(22)'))