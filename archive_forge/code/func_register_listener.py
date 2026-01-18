import logging
def register_listener(self, identifier, func, filter_func=None):
    identifier = _to_tuple(identifier)
    substrings = (identifier[:i] for i in range(1, len(identifier) + 1))
    for partial_id in substrings:
        self._listeners.setdefault(partial_id, []).append((func, filter_func))