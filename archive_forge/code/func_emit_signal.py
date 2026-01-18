import logging
def emit_signal(self, identifier, data):
    identifier = _to_tuple(identifier)
    LOG.debug('SIGNAL: %s emitted with data: %s ', identifier, data)
    for func, filter_func in self._listeners.get(identifier, []):
        if not filter_func or filter_func(data):
            func(identifier, data)