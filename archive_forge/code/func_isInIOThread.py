from functools import wraps
def isInIOThread():
    """Are we in the thread responsible for I/O requests (the event loop)?"""
    return ioThread == getThreadID()