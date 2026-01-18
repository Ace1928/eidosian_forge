from .. import tests, utextwrap
def test_fill_without_breaks(self):
    text = 'spam ham egg spamhamegg' + _str_D + ' spam' + _str_D * 2
    self.assertEqual('\n'.join(['spam ham', 'egg', 'spamhamegg', _str_D, 'spam' + _str_D[:2], _str_D[2:] + _str_D[:2], _str_D[2:]]), utextwrap.fill(text, 8, break_long_words=False))