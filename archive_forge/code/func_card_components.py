import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
@pytest.fixture
def card_components():
    w1 = FloatSlider(name='Slider', css_classes=['class_w1'])
    w2 = TextInput(name='Text:', css_classes=['class_w2'])
    return (w1, w2)