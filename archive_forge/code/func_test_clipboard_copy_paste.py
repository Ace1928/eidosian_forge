from kivy.tests.common import GraphicUnitTest
def test_clipboard_copy_paste(self):
    clippy = self._clippy
    txt1 = u'Hello 1'
    clippy.copy(txt1)
    ret = clippy.paste()
    self.assertEqual(txt1, ret)