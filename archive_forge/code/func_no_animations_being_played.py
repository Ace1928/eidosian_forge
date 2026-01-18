import pytest
def no_animations_being_played():
    from kivy.animation import Animation
    return len(Animation._instances) == 0