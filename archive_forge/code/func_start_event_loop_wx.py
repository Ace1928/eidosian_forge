from IPython.core.getipython import get_ipython
def start_event_loop_wx(app=None):
    """Start the wx event loop in a consistent manner."""
    if app is None:
        app = get_app_wx()
    if not is_event_loop_running_wx(app):
        app._in_event_loop = True
        app.MainLoop()
        app._in_event_loop = False
    else:
        app._in_event_loop = True