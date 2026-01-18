from .. import tests, utextwrap
def test_ambiwidth1(self):
    w = utextwrap.UTextWrapper(4, ambiguous_width=1)
    s = self._cyrill_char * 8
    self.assertEqual([self._cyrill_char * 4] * 2, w.wrap(s))