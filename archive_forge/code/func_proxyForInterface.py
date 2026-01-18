from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def proxyForInterface(iface, originalAttribute='original'):
    """
    Create a class which proxies all method calls which adhere to an interface
    to another provider of that interface.

    This function is intended for creating specialized proxies. The typical way
    to use it is by subclassing the result::

      class MySpecializedProxy(proxyForInterface(IFoo)):
          def someInterfaceMethod(self, arg):
              if arg == 3:
                  return 3
              return self.original.someInterfaceMethod(arg)

    @param iface: The Interface to which the resulting object will conform, and
        which the wrapped object must provide.

    @param originalAttribute: name of the attribute used to save the original
        object in the resulting class. Default to C{original}.
    @type originalAttribute: C{str}

    @return: A class whose constructor takes the original object as its only
        argument. Constructing the class creates the proxy.
    """

    def __init__(self, original):
        setattr(self, originalAttribute, original)
    contents: Dict[str, object] = {'__init__': __init__}
    for name in iface:
        contents[name] = _ProxyDescriptor(name, originalAttribute)
    proxy = type(f'(Proxy for {reflect.qual(iface)})', (object,), contents)
    declarations.classImplements(proxy, iface)
    return proxy