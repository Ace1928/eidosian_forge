import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def extension_add_method(self, object, name, function):
    """extension_add_method(object, name, function)

        Add an X extension module method.  OBJECT is the type of
        object to add the function to, a string from this list:

            display
            resource
            drawable
            window
            pixmap
            fontable
            font
            gc
            colormap
            cursor

        NAME is the name of the method, a string.  FUNCTION is a
        normal function whose first argument is a 'self'.
        """
    if object == 'display':
        if hasattr(self, name):
            raise AssertionError('attempting to replace display method: %s' % name)
        self.display_extension_methods[name] = function
    else:
        types = (object,) + _resource_hierarchy.get(object, ())
        for type in types:
            cls = _resource_baseclasses[type]
            if hasattr(cls, name):
                raise AssertionError('attempting to replace %s method: %s' % (type, name))
            try:
                self.class_extension_dicts[type][name] = function
            except KeyError:
                self.class_extension_dicts[type] = {name: function}