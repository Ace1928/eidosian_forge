from ._version import __version__, __protocol_version__, __jupyter_widgets_controls_version__, __jupyter_widgets_base_version__
import os
import sys
from traitlets import link, dlink
from IPython import get_ipython
from .widgets import *
def register_comm_target(kernel=None):
    """Register the jupyter.widget comm target"""
    from . import comm
    comm_manager = comm.get_comm_manager()
    if comm_manager is None:
        return
    comm_manager.register_target('jupyter.widget', Widget.handle_comm_opened)
    comm_manager.register_target('jupyter.widget.control', Widget.handle_control_comm_opened)