import threading
def convert_to_timeout(value=None, default_value=None, event_factory=threading.Event):
    """Converts a given value to a timeout instance (and returns it).

    Does nothing if the value provided is already a timeout instance.
    """
    if value is None:
        value = default_value
    if isinstance(value, (int, float, str)):
        return Timeout(float(value), event_factory=event_factory)
    elif isinstance(value, Timeout):
        return value
    else:
        raise ValueError("Invalid timeout literal '%s'" % value)