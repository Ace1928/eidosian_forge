from a single quote by the algorithm. Therefore, a text like::
import re, sys
def test_dates(self):
    self.assertEqual(smartyPants("1440-80's"), '1440-80’s')
    self.assertEqual(smartyPants("1440-'80s"), '1440-’80s')
    self.assertEqual(smartyPants("1440---'80s"), '1440–’80s')
    self.assertEqual(smartyPants("1960's"), '1960’s')
    self.assertEqual(smartyPants("one two '60s"), 'one two ’60s')
    self.assertEqual(smartyPants("'60s"), '’60s')