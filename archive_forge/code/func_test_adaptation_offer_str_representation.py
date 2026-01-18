import sys
import unittest
from traits.adaptation.adaptation_offer import AdaptationOffer
def test_adaptation_offer_str_representation(self):
    """ test string representation of the AdaptationOffer class. """

    class Foo:
        pass

    class Bar:
        pass
    adaptation_offer = AdaptationOffer(from_protocol=Foo, to_protocol=Bar)
    desired_repr = "<AdaptationOffer: '{}' -> '{}'>".format(adaptation_offer.from_protocol_name, adaptation_offer.to_protocol_name)
    self.assertEqual(desired_repr, str(adaptation_offer))
    self.assertEqual(desired_repr, repr(adaptation_offer))