import inspect
import logging
import urllib.parse
def inspect_kwargs(kallable):
    try:
        signature = inspect.signature(kallable)
    except AttributeError:
        try:
            args, varargs, keywords, defaults = inspect.getargspec(kallable)
        except TypeError:
            return {}
        if not defaults:
            return {}
        supported_keywords = args[-len(defaults):]
        return dict(zip(supported_keywords, defaults))
    else:
        return {name: param.default for name, param in signature.parameters.items() if param.default != inspect.Parameter.empty}