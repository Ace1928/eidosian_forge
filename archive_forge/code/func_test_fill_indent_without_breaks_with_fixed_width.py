from .. import tests, utextwrap
def test_fill_indent_without_breaks_with_fixed_width(self):
    w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
    w.break_long_words = False
    w.width = 3
    self.assertEqual('\n'.join(['    hello', '    ' + _str_D[0], '    ' + _str_D[1], '    ' + _str_D[2], '    ' + _str_D[3]]), w.fill(_str_SD))