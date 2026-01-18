import sys
def registerDOMImplementation(name, factory):
    """registerDOMImplementation(name, factory)

    Register the factory function with the name. The factory function
    should return an object which implements the DOMImplementation
    interface. The factory function can either return the same object,
    or a new one (e.g. if that implementation supports some
    customization)."""
    registered[name] = factory