import collections
import inspect
from automat import MethodicalMachine
from twisted.python.modules import PythonModule, getModule
def isOriginalLocation(attr):
    """
    Attempt to discover if this appearance of a PythonAttribute
    representing a class refers to the module where that class was
    defined.
    """
    sourceModule = inspect.getmodule(attr.load())
    if sourceModule is None:
        return False
    currentModule = attr
    while not isinstance(currentModule, PythonModule):
        currentModule = currentModule.onObject
    return currentModule.name == sourceModule.__name__