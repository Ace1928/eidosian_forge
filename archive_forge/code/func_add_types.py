import os, sys
from OpenGL.plugins import PlatformPlugin
from OpenGL import _configflags
def add_types(function):
    """Adds the given metadata to the function, introspects var names from declaration"""
    function.resultType = resultType
    function.argTypes = argTypes
    if hasattr(function, 'func_code'):
        function.argNames = function.func_code.co_varnames
    else:
        function.argNames = function.__code__.co_varnames
    if _configflags.TYPE_ANNOTATIONS:
        function.__annotations__ = {'return': resultType}
        for name, typ in zip(function.argNames, argTypes):
            function.__annotations__[name] = typ
    return function