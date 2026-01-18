def dcite(self, *args, **kwargs):
    """If I could cite I would"""

    def nondecorating_decorator(func):
        return func
    return nondecorating_decorator