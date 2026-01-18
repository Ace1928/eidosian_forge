import paste.util.threadinglocal as threadinglocal
def restoration_begin(self, request_id):
    """Enable a restoration context in the current thread for the specified
        request_id"""
    if request_id in self.saved_registry_states:
        registry, reglist = self.saved_registry_states[request_id]
        registry.reglist = reglist
    self.restoration_context_id.request_id = request_id