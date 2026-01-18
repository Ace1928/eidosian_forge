from __future__ import annotations
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from ..key_bindings import KeyBindings
def load_cpr_bindings() -> KeyBindings:
    key_bindings = KeyBindings()

    @key_bindings.add(Keys.CPRResponse, save_before=lambda e: False)
    def _(event: E) -> None:
        """
        Handle incoming Cursor-Position-Request response.
        """
        row, col = map(int, event.data[2:-1].split(';'))
        event.app.renderer.report_absolute_cursor_row(row)
    return key_bindings