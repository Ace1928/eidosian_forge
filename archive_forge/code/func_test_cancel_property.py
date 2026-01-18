import pytest
def test_cancel_property(self):
    from kivy.animation import Animation
    from kivy.uix.widget import Widget
    a = Animation(x=100) & Animation(y=100)
    w = Widget()
    a.start(w)
    a.cancel_property(w, 'x')
    assert not no_animations_being_played()
    a.stop(w)
    assert no_animations_being_played()