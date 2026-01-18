from OpenGL import platform
import weakref
def delValue(constant, context=None):
    """Delete the specified value for the given context
    
    constant -- Normally a GL constant value, but can be any hashable value 
    context -- the context identifier for which we're storing the value
    """
    context = getContext(context)
    found = False
    for storage in STORAGES:
        contextStorage = storage.get(context)
        if contextStorage:
            try:
                del contextStorage[constant]
                found = True
            except KeyError as err:
                pass
    return found