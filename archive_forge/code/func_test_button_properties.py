from kivy.tests.common import GraphicUnitTest
def test_button_properties(self):
    from kivy.uix.button import Button
    r = self.render
    r(Button(text='Hello world'))
    r(Button(text='Multiline\ntext\nbutton'))
    r(Button(text='Hello world', font_size=42))
    r(Button(text='This is my first line\nSecond line', halign='center'))