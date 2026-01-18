from contextlib import contextmanager
from abc import ABC
from abc import abstractmethod
def get_remote_obj(self):
    global _current_server
    real_ref_id = self.get_real_ref_from_server()
    if real_ref_id is None:
        return None
    return _current_server.lookup_or_register_func(real_ref_id, self.client_id, None)