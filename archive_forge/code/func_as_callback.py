from functools import wraps
@classmethod
def as_callback(cls, maybe_callback=None):
    """Transform callback=... into Callback instance

        For the special value of ``None``, return the global instance of
        ``NoOpCallback``. This is an alternative to including
        ``callback=DEFAULT_CALLBACK`` directly in a method signature.
        """
    if maybe_callback is None:
        return DEFAULT_CALLBACK
    return maybe_callback