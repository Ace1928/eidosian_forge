from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def request_keyboard(self, callback, target, input_type='text', keyboard_suggestions=True):
    """.. versionadded:: 1.0.4

        Internal widget method to request the keyboard. This method is rarely
        required by the end-user as it is handled automatically by the
        :class:`~kivy.uix.textinput.TextInput`. We expose it in case you want
        to handle the keyboard manually for unique input scenarios.

        A widget can request the keyboard, indicating a callback to call
        when the keyboard is released (or taken by another widget).

        :Parameters:
            `callback`: func
                Callback that will be called when the keyboard is
                closed. This can be because somebody else requested the
                keyboard or the user closed it.
            `target`: Widget
                Attach the keyboard to the specified `target`. This should be
                the widget that requested the keyboard. Ensure you have a
                different target attached to each keyboard if you're working in
                a multi user mode.

                .. versionadded:: 1.0.8

            `input_type`: string
                Choose the type of soft keyboard to request. Can be one of
                'null', 'text', 'number', 'url', 'mail', 'datetime', 'tel',
                'address'.

                .. note::

                    `input_type` is currently only honored on Android.

                .. versionadded:: 1.8.0

                .. versionchanged:: 2.1.0
                    Added `null` to soft keyboard types.

            `keyboard_suggestions`: bool
                If True provides auto suggestions on top of keyboard.
                This will only work if input_type is set to `text`, `url`,
                `mail` or `address`.

                .. versionadded:: 2.1.0

        :Return:
            An instance of :class:`Keyboard` containing the callback, target,
            and if the configuration allows it, a
            :class:`~kivy.uix.vkeyboard.VKeyboard` instance attached as a
            *.widget* property.

        .. note::

            The behavior of this function is heavily influenced by the current
            `keyboard_mode`. Please see the Config's
            :ref:`configuration tokens <configuration-tokens>` section for
            more information.

        """
    self.release_keyboard(target)
    if self.allow_vkeyboard:
        keyboard = None
        global VKeyboard
        if VKeyboard is None and self._vkeyboard_cls is None:
            from kivy.uix.vkeyboard import VKeyboard
            self._vkeyboard_cls = VKeyboard
        key = 'single' if self.single_vkeyboard else target
        if key not in self._keyboards:
            vkeyboard = self._vkeyboard_cls()
            keyboard = Keyboard(widget=vkeyboard, window=self)
            vkeyboard.bind(on_key_down=keyboard._on_vkeyboard_key_down, on_key_up=keyboard._on_vkeyboard_key_up, on_textinput=keyboard._on_vkeyboard_textinput)
            self._keyboards[key] = keyboard
        else:
            keyboard = self._keyboards[key]
        keyboard.target = keyboard.widget.target = target
        keyboard.callback = keyboard.widget.callback = callback
        self.add_widget(keyboard.widget)
        keyboard.widget.docked = self.docked_vkeyboard
        keyboard.widget.setup_mode()
        if self.softinput_mode == 'pan':
            keyboard.widget.top = 0
        elif self.softinput_mode == 'below_target':
            keyboard.widget.top = keyboard.target.y
    else:
        keyboard = self._system_keyboard
        keyboard.callback = callback
        keyboard.target = target
    if self.allow_vkeyboard and self.use_syskeyboard:
        self.unbind(on_key_down=keyboard._on_window_key_down, on_key_up=keyboard._on_window_key_up, on_textinput=keyboard._on_window_textinput)
        self.bind(on_key_down=keyboard._on_window_key_down, on_key_up=keyboard._on_window_key_up, on_textinput=keyboard._on_window_textinput)
    return keyboard