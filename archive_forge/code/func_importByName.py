import logging 
def importByName(fullName):
    """Import a class by name"""
    name = fullName.split('.')
    moduleName = name[:-1]
    className = name[-1]
    module = __import__('.'.join(moduleName), {}, {}, moduleName)
    return getattr(module, className)