from kivy.tests.common import GraphicUnitTest
def test_boxlayout_padding_spacing(self):
    from kivy.uix.boxlayout import BoxLayout
    r = self.render
    b = self.box
    layout = BoxLayout(spacing=20, padding=20)
    layout.add_widget(b(1, 0, 0))
    layout.add_widget(b(0, 1, 0))
    layout.add_widget(b(0, 0, 1))
    r(layout)
    layout = BoxLayout(spacing=20, padding=20, orientation='vertical')
    layout.add_widget(b(1, 0, 0))
    layout.add_widget(b(0, 1, 0))
    layout.add_widget(b(0, 0, 1))
    r(layout)