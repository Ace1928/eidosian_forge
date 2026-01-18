from kivy.tests import async_run, UnitKivyApp
def dropdown_app():
    from kivy.app import App
    from kivy.uix.button import Button
    from kivy.uix.dropdown import DropDown
    from kivy.uix.label import Label

    class RootButton(Button):
        dropdown = None

        def on_touch_down(self, touch):
            assert self.dropdown.attach_to is None
            return super(RootButton, self).on_touch_down(touch)

        def on_touch_move(self, touch):
            assert self.dropdown.attach_to is None
            return super(RootButton, self).on_touch_move(touch)

        def on_touch_up(self, touch):
            assert self.dropdown.attach_to is None
            return super(RootButton, self).on_touch_up(touch)

    class TestApp(UnitKivyApp, App):

        def build(self):
            root = RootButton(text='Root')
            self.attach_widget = Label(text='Attached widget')
            root.add_widget(self.attach_widget)
            root.dropdown = self.dropdown = DropDown(auto_dismiss=True, min_state_time=0)
            self.inner_widget = w = Label(size_hint=(None, None), text='Dropdown')
            root.dropdown.add_widget(w)
            return root
    return TestApp()