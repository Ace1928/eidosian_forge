from __future__ import annotations
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from .containers import Window
from .controls import FormattedTextControl
from .dimension import D
from .layout import Layout
def create_dummy_layout() -> Layout:
    """
    Create a dummy layout for use in an 'Application' that doesn't have a
    layout specified. When ENTER is pressed, the application quits.
    """
    kb = KeyBindings()

    @kb.add('enter')
    def enter(event: E) -> None:
        event.app.exit()
    control = FormattedTextControl(HTML('No layout specified. Press <reverse>ENTER</reverse> to quit.'), key_bindings=kb)
    window = Window(content=control, height=D(min=1))
    return Layout(container=window, focused_element=window)