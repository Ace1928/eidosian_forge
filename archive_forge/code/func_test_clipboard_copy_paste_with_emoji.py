from kivy.tests.common import GraphicUnitTest
def test_clipboard_copy_paste_with_emoji(self):
    clippy = self._clippy
    test_emoji_str = 'kivy 😀 😁 🤣 😃 😄 😅 😆 😉 😊 😋 😎 😍 😘 😗'
    clippy.copy(test_emoji_str)
    self.assertEqual(test_emoji_str, clippy.paste())