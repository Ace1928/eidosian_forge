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
def set_system_cursor(self, cursor_name):
    """Set type of a mouse cursor in the Window.

        It can be one of 'arrow', 'ibeam', 'wait', 'crosshair', 'wait_arrow',
        'size_nwse', 'size_nesw', 'size_we', 'size_ns', 'size_all', 'no', or
        'hand'.

        On some platforms there might not be a specific cursor supported and
        such an option falls back to one of the substitutable alternatives:

        +------------+-----------+------------+-----------+---------------+
        |            | Windows   | MacOS      | Linux X11 | Linux Wayland |
        +============+===========+============+===========+===============+
        | arrow      | arrow     | arrow      | arrow     | arrow         |
        +------------+-----------+------------+-----------+---------------+
        | ibeam      | ibeam     | ibeam      | ibeam     | ibeam         |
        +------------+-----------+------------+-----------+---------------+
        | wait       | wait      | arrow      | wait      | wait          |
        +------------+-----------+------------+-----------+---------------+
        | crosshair  | crosshair | crosshair  | crosshair | hand          |
        +------------+-----------+------------+-----------+---------------+
        | wait_arrow | arrow     | arrow      | wait      | wait          |
        +------------+-----------+------------+-----------+---------------+
        | size_nwse  | size_nwse | size_all   | size_all  | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_nesw  | size_nesw | size_all   | size_all  | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_we    | size_we   | size_we    | size_we   | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_ns    | size_ns   | size_ns    | size_ns   | hand          |
        +------------+-----------+------------+-----------+---------------+
        | size_all   | size_all  | size_all   | size_all  | hand          |
        +------------+-----------+------------+-----------+---------------+
        | no         | no        | no         | no        | ibeam         |
        +------------+-----------+------------+-----------+---------------+
        | hand       | hand      | hand       | hand      | hand          |
        +------------+-----------+------------+-----------+---------------+

        .. versionadded:: 1.10.1

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
    pass