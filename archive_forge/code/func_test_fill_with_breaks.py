from .. import tests, utextwrap
def test_fill_with_breaks(self):
    text = 'spam ham egg spamhamegg' + _str_D + ' spam' + _str_D * 2
    self.assertEqual('\n'.join(['spam ham', 'egg spam', 'hamegg' + _str_D[0], _str_D[1:], 'spam' + _str_D[:2], _str_D[2:] + _str_D[:2], _str_D[2:]]), utextwrap.fill(text, 8))