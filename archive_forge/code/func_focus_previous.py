from __future__ import annotations
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
def focus_previous(event: E) -> None:
    """
    Focus the previous visible Window.
    (Often bound to the `BackTab` key.)
    """
    event.app.layout.focus_previous()