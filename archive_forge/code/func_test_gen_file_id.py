from ... import tests
from .. import generate_ids
def test_gen_file_id(self):
    gen_file_id = generate_ids.gen_file_id
    self.assertStartsWith(gen_file_id('bar'), b'bar-')
    self.assertStartsWith(gen_file_id('Mwoo oof\t m'), b'mwoooofm-')
    self.assertStartsWith(gen_file_id('..gam.py'), b'gam.py-')
    self.assertStartsWith(gen_file_id('..Mwoo oof\t m'), b'mwoooofm-')
    self.assertStartsWith(gen_file_id('åµ.txt'), b'txt-')
    fid = gen_file_id('A' * 50 + '.txt')
    self.assertStartsWith(fid, b'a' * 20 + b'-')
    self.assertTrue(len(fid) < 60)
    fid = gen_file_id('åµ..aBcd\tefGhijKLMnop\tqrstuvwxyz')
    self.assertStartsWith(fid, b'abcdefghijklmnopqrst-')
    self.assertTrue(len(fid) < 60)