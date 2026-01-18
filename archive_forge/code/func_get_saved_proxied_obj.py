import paste.util.threadinglocal as threadinglocal
def get_saved_proxied_obj(self, stacked, request_id):
    """Retrieve the saved object proxied by the specified
        StackedObjectProxy for the request identified by request_id"""
    reglist = self.saved_registry_states[request_id][1]
    stack_level = len(reglist) - 1
    stacked_id = id(stacked)
    while True:
        if stack_level < 0:
            return stacked._current_obj_orig()
        context = reglist[stack_level]
        if stacked_id in context:
            break
        stack_level -= 1
    return context[stacked_id][1]