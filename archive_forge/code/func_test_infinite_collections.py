from yaql.language import exceptions
import yaql.tests
def test_infinite_collections(self):
    self.assertRaises(exceptions.CollectionTooLargeException, self.eval, 'len(list(sequence()))')
    self.assertRaises(exceptions.CollectionTooLargeException, self.eval, 'list(sequence())')
    self.assertRaises(exceptions.CollectionTooLargeException, self.eval, 'len(dict(sequence().select([$, $])))')
    self.assertRaises(exceptions.CollectionTooLargeException, self.eval, 'dict(sequence().select([$, $]))')
    self.assertRaises(exceptions.CollectionTooLargeException, self.eval, 'sequence()')
    self.assertRaises(exceptions.CollectionTooLargeException, self.eval, 'set(sequence())')