import paste.util.threadinglocal as threadinglocal
def restoration_end(self):
    """Register a restoration context as finished, if one exists"""
    try:
        del self.restoration_context_id.request_id
    except AttributeError:
        pass