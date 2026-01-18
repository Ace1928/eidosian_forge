import pytest
def test_animated_properties(self):
    from kivy.animation import Animation
    a = Animation(x=100) & Animation(y=100)
    assert a.animated_properties == {'x': 100, 'y': 100}