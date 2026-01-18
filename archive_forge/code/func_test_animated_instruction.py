import pytest
def test_animated_instruction(self):
    from kivy.graphics import Scale
    from kivy.animation import Animation
    a = Animation(x=100, d=1)
    instruction = Scale(3, 3, 3)
    a.start(instruction)
    assert a.animated_properties == {'x': 100}
    assert instruction.x == pytest.approx(3)
    sleep(1.5)
    assert instruction.x == pytest.approx(100)
    assert no_animations_being_played()