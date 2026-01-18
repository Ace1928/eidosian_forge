from kivy.tests.common import GraphicUnitTest
def test_anchorlayout_x(self):
    from kivy.uix.anchorlayout import AnchorLayout
    r = self.render
    b = self.box
    layout = AnchorLayout(anchor_x='left')
    layout.add_widget(b(1, 0, 0))
    r(layout)
    layout = AnchorLayout(anchor_x='center')
    layout.add_widget(b(1, 0, 0))
    r(layout)
    layout = AnchorLayout(anchor_x='right')
    layout.add_widget(b(1, 0, 0))
    r(layout)