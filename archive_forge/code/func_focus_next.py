from __future__ import annotations
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
def focus_next(event: E) -> None:
    """
    Focus the next visible Window.
    (Often bound to the `Tab` key.)
    """
    event.app.layout.focus_next()