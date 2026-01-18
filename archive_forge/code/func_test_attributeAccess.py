import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
def test_attributeAccess(self):
    """
        A sentence attribute gets the correct value, and accessing an
        unset attribute (which is specified as being a valid sentence
        attribute) gets L{None}.
        """
    thisSentinel = object()
    sentence = self.sentenceClass({sentinelValueOne: thisSentinel})
    self.assertEqual(getattr(sentence, sentinelValueOne), thisSentinel)
    self.assertIsNone(getattr(sentence, sentinelValueTwo))