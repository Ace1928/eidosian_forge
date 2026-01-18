import unittest
import idna
def test_check_bidi(self):
    l = 'a'
    r = 'א'
    al = 'ا'
    an = '٠'
    en = '0'
    es = '-'
    cs = ','
    et = '$'
    on = '!'
    bn = '\u200c'
    nsm = 'ؐ'
    ws = ' '
    self.assertTrue(idna.check_bidi(l))
    self.assertTrue(idna.check_bidi(r))
    self.assertTrue(idna.check_bidi(al))
    self.assertRaises(idna.IDNABidiError, idna.check_bidi, an)
    self.assertTrue(idna.check_bidi(r + al))
    self.assertTrue(idna.check_bidi(r + al))
    self.assertTrue(idna.check_bidi(r + an))
    self.assertTrue(idna.check_bidi(r + en))
    self.assertTrue(idna.check_bidi(r + es + al))
    self.assertTrue(idna.check_bidi(r + cs + al))
    self.assertTrue(idna.check_bidi(r + et + al))
    self.assertTrue(idna.check_bidi(r + on + al))
    self.assertTrue(idna.check_bidi(r + bn + al))
    self.assertTrue(idna.check_bidi(r + nsm))
    self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + l)
    self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + ws)
    self.assertTrue(idna.check_bidi(r + al))
    self.assertTrue(idna.check_bidi(r + en))
    self.assertTrue(idna.check_bidi(r + an))
    self.assertTrue(idna.check_bidi(r + nsm))
    self.assertTrue(idna.check_bidi(r + nsm + nsm))
    self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + on)
    self.assertTrue(idna.check_bidi(r + en))
    self.assertTrue(idna.check_bidi(r + an))
    self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + en + an)
    self.assertRaises(idna.IDNABidiError, idna.check_bidi, r + an + en)
    self.assertTrue(idna.check_bidi(l + en, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + es + l, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + cs + l, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + et + l, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + on + l, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + bn + l, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + nsm, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + l, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + en, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + en + nsm, check_ltr=True))
    self.assertTrue(idna.check_bidi(l + en + nsm + nsm, check_ltr=True))
    self.assertRaises(idna.IDNABidiError, idna.check_bidi, l + cs, check_ltr=True)