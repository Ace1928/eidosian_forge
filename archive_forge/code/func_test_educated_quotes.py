from a single quote by the algorithm. Therefore, a text like::
import re, sys
def test_educated_quotes(self):
    self.assertEqual(smartyPants('"Isn\'t this fun?"'), '“Isn’t this fun?”')