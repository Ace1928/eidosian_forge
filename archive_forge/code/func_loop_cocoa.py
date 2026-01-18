import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@register_integration('osx')
def loop_cocoa(kernel):
    """Start the kernel, coordinating with the Cocoa CFRunLoop event loop
    via the matplotlib MacOSX backend.
    """
    from ._eventloop_macos import mainloop, stop
    real_excepthook = sys.excepthook

    def handle_int(etype, value, tb):
        """don't let KeyboardInterrupts look like crashes"""
        stop()
        if etype is KeyboardInterrupt:
            print('KeyboardInterrupt caught in CFRunLoop', file=sys.__stdout__)
        else:
            real_excepthook(etype, value, tb)
    while not kernel.shell.exit_now:
        try:
            try:
                sys.excepthook = handle_int
                mainloop(kernel._poll_interval)
                if kernel.shell_stream.flush(limit=1):
                    return
            except BaseException:
                raise
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught in kernel', file=sys.__stdout__)
        finally:
            sys.excepthook = real_excepthook