from .. import tests, utextwrap
def check_cut(self, text, width, pos):
    w = utextwrap.UTextWrapper()
    self.assertEqual((text[:pos], text[pos:]), w._cut(text, width))