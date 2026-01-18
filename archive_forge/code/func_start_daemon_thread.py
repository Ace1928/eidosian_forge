import logging
import threading
def start_daemon_thread(*args, **kwargs):
    """Starts a thread and marks it as a daemon thread."""
    thread = threading.Thread(*args, **kwargs)
    thread.daemon = True
    thread.start()
    return thread