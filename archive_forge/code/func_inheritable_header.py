from typing import Optional, TypeVar
def inheritable_header(text):

    def _wrapped(cls):
        setattr(cls, _INHERITABLE_HEADER, text)
        return cls
    return _wrapped