from kivy.tests.common import GraphicUnitTest
def test_add_widget_index_2(self):
    from kivy.uix.widget import Widget
    from kivy.uix.button import Button
    r = self.render
    root = Widget()
    a = Button(text='Hello')
    b = Button(text='World', pos=(50, 10))
    c = Button(text='Kivy', pos=(10, 50))
    root.add_widget(a)
    root.add_widget(b)
    root.add_widget(c, 2)
    r(root)