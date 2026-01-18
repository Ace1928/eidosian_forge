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
def configure_keyboards(self):
    sk = self._system_keyboard
    self.bind(on_key_down=sk._on_window_key_down, on_key_up=sk._on_window_key_up, on_textinput=sk._on_window_textinput)
    self.use_syskeyboard = True
    self.allow_vkeyboard = False
    self.single_vkeyboard = True
    self.docked_vkeyboard = False
    mode = Config.get('kivy', 'keyboard_mode')
    if mode not in ('', 'system', 'dock', 'multi', 'systemanddock', 'systemandmulti'):
        Logger.critical('Window: unknown keyboard mode %r' % mode)
    if mode == 'system':
        self.use_syskeyboard = True
        self.allow_vkeyboard = False
        self.single_vkeyboard = True
        self.docked_vkeyboard = False
    elif mode == 'dock':
        self.use_syskeyboard = False
        self.allow_vkeyboard = True
        self.single_vkeyboard = True
        self.docked_vkeyboard = True
    elif mode == 'multi':
        self.use_syskeyboard = False
        self.allow_vkeyboard = True
        self.single_vkeyboard = False
        self.docked_vkeyboard = False
    elif mode == 'systemanddock':
        self.use_syskeyboard = True
        self.allow_vkeyboard = True
        self.single_vkeyboard = True
        self.docked_vkeyboard = True
    elif mode == 'systemandmulti':
        self.use_syskeyboard = True
        self.allow_vkeyboard = True
        self.single_vkeyboard = False
        self.docked_vkeyboard = False
    Logger.info('Window: virtual keyboard %sallowed, %s, %s' % ('' if self.allow_vkeyboard else 'not ', 'single mode' if self.single_vkeyboard else 'multiuser mode', 'docked' if self.docked_vkeyboard else 'not docked'))