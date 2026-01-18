from __future__ import unicode_literals
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.token import Token
def find_window_for_buffer_name(cli, buffer_name):
    """
    Look for a :class:`~prompt_toolkit.layout.containers.Window` in the Layout
    that contains the :class:`~prompt_toolkit.layout.controls.BufferControl`
    for the given buffer and return it. If no such Window is found, return None.
    """
    from prompt_toolkit.interface import CommandLineInterface
    assert isinstance(cli, CommandLineInterface)
    from .containers import Window
    from .controls import BufferControl
    for l in cli.layout.walk(cli):
        if isinstance(l, Window) and isinstance(l.content, BufferControl):
            if l.content.buffer_name == buffer_name:
                return l