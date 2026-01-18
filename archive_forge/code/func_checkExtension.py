import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def checkExtension(self, name):
    """Check whether the given extension is supported by current context"""
    if not name:
        return True
    context = self.GetCurrentContext()
    if context:
        from OpenGL import contextdata
        set = contextdata.getValue('extensions', context=context)
        if set is None:
            set = {}
            contextdata.setValue('extensions', set, context=context, weak=False)
        current = set.get(name)
        if current is None:
            from OpenGL import extensions
            result = extensions.ExtensionQuerier.hasExtension(name)
            set[name] = result
            return result
        return current
    else:
        from OpenGL import extensions
        return extensions.ExtensionQuerier.hasExtension(name)