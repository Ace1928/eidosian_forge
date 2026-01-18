from yaql.language import exceptions
import yaql.tests
def test_characters(self):
    self.assertCountEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], self.eval('characters(octdigits => true, digits => true)'))