from kivy.tests.common import GraphicUnitTest
def test_clipboard_copy(self):
    clippy = self._clippy
    try:
        clippy.copy(u'Hello World')
    except:
        self.fail('Can not put data to clipboard')