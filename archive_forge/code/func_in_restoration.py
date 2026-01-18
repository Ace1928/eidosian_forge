import paste.util.threadinglocal as threadinglocal
def in_restoration(self):
    """Determine if a restoration context is active for the current thread.
        Returns the request_id it's active for if so, otherwise False"""
    return getattr(self.restoration_context_id, 'request_id', False)