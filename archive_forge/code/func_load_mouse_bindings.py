from __future__ import unicode_literals
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import HasSelection, Condition, EmacsInsertMode, ViInsertMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.mouse_events import MouseEventType, MouseEvent
from prompt_toolkit.renderer import HeightIsUnknownError
from prompt_toolkit.utils import suspend_to_background_supported, is_windows
from .named_commands import get_by_name
from ..registry import Registry
def load_mouse_bindings():
    """
    Key bindings, required for mouse support.
    (Mouse events enter through the key binding system.)
    """
    registry = Registry()

    @registry.add_binding(Keys.Vt100MouseEvent)
    def _(event):
        """
        Handling of incoming mouse event.
        """
        if event.data[2] == 'M':
            mouse_event, x, y = map(ord, event.data[3:])
            mouse_event = {32: MouseEventType.MOUSE_DOWN, 35: MouseEventType.MOUSE_UP, 96: MouseEventType.SCROLL_UP, 97: MouseEventType.SCROLL_DOWN}.get(mouse_event)
            if x >= 56320:
                x -= 56320
            if y >= 56320:
                y -= 56320
            x -= 32
            y -= 32
        else:
            data = event.data[2:]
            if data[:1] == '<':
                sgr = True
                data = data[1:]
            else:
                sgr = False
            mouse_event, x, y = map(int, data[:-1].split(';'))
            m = data[-1]
            if sgr:
                mouse_event = {(0, 'M'): MouseEventType.MOUSE_DOWN, (0, 'm'): MouseEventType.MOUSE_UP, (64, 'M'): MouseEventType.SCROLL_UP, (65, 'M'): MouseEventType.SCROLL_DOWN}.get((mouse_event, m))
            else:
                mouse_event = {32: MouseEventType.MOUSE_DOWN, 35: MouseEventType.MOUSE_UP, 96: MouseEventType.SCROLL_UP, 97: MouseEventType.SCROLL_DOWN}.get(mouse_event)
        x -= 1
        y -= 1
        if event.cli.renderer.height_is_known and mouse_event is not None:
            try:
                y -= event.cli.renderer.rows_above_layout
            except HeightIsUnknownError:
                return
            handler = event.cli.renderer.mouse_handlers.mouse_handlers[x, y]
            handler(event.cli, MouseEvent(position=Point(x=x, y=y), event_type=mouse_event))

    @registry.add_binding(Keys.WindowsMouseEvent)
    def _(event):
        """
        Handling of mouse events for Windows.
        """
        assert is_windows()
        event_type, x, y = event.data.split(';')
        x = int(x)
        y = int(y)
        screen_buffer_info = event.cli.renderer.output.get_win32_screen_buffer_info()
        rows_above_cursor = screen_buffer_info.dwCursorPosition.Y - event.cli.renderer._cursor_pos.y
        y -= rows_above_cursor
        handler = event.cli.renderer.mouse_handlers.mouse_handlers[x, y]
        handler(event.cli, MouseEvent(position=Point(x=x, y=y), event_type=event_type))
    return registry