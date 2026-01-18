from kivy.tests.common import GraphicUnitTest
def test_clipboard_paste(self):
    clippy = self._clippy
    try:
        clippy.paste()
    except:
        self.fail('Can not get data from clipboard')