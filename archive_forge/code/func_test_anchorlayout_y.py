from kivy.tests.common import GraphicUnitTest
def test_anchorlayout_y(self):
    from kivy.uix.anchorlayout import AnchorLayout
    r = self.render
    b = self.box
    layout = AnchorLayout(anchor_y='bottom')
    layout.add_widget(b(1, 0, 0))
    r(layout)
    layout = AnchorLayout(anchor_y='center')
    layout.add_widget(b(1, 0, 0))
    r(layout)
    layout = AnchorLayout(anchor_y='top')
    layout.add_widget(b(1, 0, 0))
    r(layout)