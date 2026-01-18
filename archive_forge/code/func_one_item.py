from __future__ import annotations
from typing import Callable, Iterable, Sequence
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text.base import OneStyleAndTextTuple, StyleAndTextTuples
from prompt_toolkit.key_binding.key_bindings import KeyBindings, KeyBindingsBase
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Shadow
from .base import Border
def one_item(i: int, item: MenuItem) -> Iterable[OneStyleAndTextTuple]:

    def mouse_handler(mouse_event: MouseEvent) -> None:
        if item.disabled:
            return
        hover = mouse_event.event_type == MouseEventType.MOUSE_MOVE
        if mouse_event.event_type == MouseEventType.MOUSE_UP or hover:
            app = get_app()
            if not hover and item.handler:
                app.layout.focus_last()
                item.handler()
            else:
                self.selected_menu = self.selected_menu[:level + 1] + [i]
    if i == selected_item:
        yield ('[SetCursorPosition]', '')
        style = 'class:menu-bar.selected-item'
    else:
        style = ''
    yield ('class:menu', Border.VERTICAL)
    if item.text == '-':
        yield (style + 'class:menu-border', f'{Border.HORIZONTAL * (menu.width + 3)}', mouse_handler)
    else:
        yield (style, f' {item.text}'.ljust(menu.width + 3), mouse_handler)
    if item.children:
        yield (style, '>', mouse_handler)
    else:
        yield (style, ' ', mouse_handler)
    if i == selected_item:
        yield ('[SetMenuPosition]', '')
    yield ('class:menu', Border.VERTICAL)
    yield ('', '\n')