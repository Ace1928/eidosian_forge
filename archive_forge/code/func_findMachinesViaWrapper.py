import collections
import inspect
from automat import MethodicalMachine
from twisted.python.modules import PythonModule, getModule
def findMachinesViaWrapper(within):
    """
    Recursively yield L{MethodicalMachine}s and their FQPNs within a
    L{PythonModule} or a L{twisted.python.modules.PythonAttribute}
    wrapper object.

    Note that L{PythonModule}s may refer to packages, as well.

    The discovery heuristic considers L{MethodicalMachine} instances
    that are module-level attributes or class-level attributes
    accessible from module scope.  Machines inside nested classes will
    be discovered, but those returned from functions or methods will not be.

    @type within: L{PythonModule} or L{twisted.python.modules.PythonAttribute}
    @param within: Where to start the search.

    @return: a generator which yields FQPN, L{MethodicalMachine} pairs.
    """
    queue = collections.deque([within])
    visited = set()
    while queue:
        attr = queue.pop()
        value = attr.load()
        if isinstance(value, MethodicalMachine) and value not in visited:
            visited.add(value)
            yield (attr.name, value)
        elif inspect.isclass(value) and isOriginalLocation(attr) and (value not in visited):
            visited.add(value)
            queue.extendleft(attr.iterAttributes())
        elif isinstance(attr, PythonModule) and value not in visited:
            visited.add(value)
            queue.extendleft(attr.iterAttributes())
            queue.extendleft(attr.iterModules())