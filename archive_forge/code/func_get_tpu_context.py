import contextlib
import threading
def get_tpu_context():
    return _current_tpu_context