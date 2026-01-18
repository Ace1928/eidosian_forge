import urllib.parse
from cherrypy._cpcompat import text_or_bytes
import cherrypy
def popargs(*args, **kwargs):
    """Decorate _cp_dispatch.

    (cherrypy.dispatch.Dispatcher.dispatch_method_name)

    Optional keyword argument: handler=(Object or Function)

    Provides a _cp_dispatch function that pops off path segments into
    cherrypy.request.params under the names specified.  The dispatch
    is then forwarded on to the next vpath element.

    Note that any existing (and exposed) member function of the class that
    popargs is applied to will override that value of the argument.  For
    instance, if you have a method named "list" on the class decorated with
    popargs, then accessing "/list" will call that function instead of popping
    it off as the requested parameter.  This restriction applies to all
    _cp_dispatch functions.  The only way around this restriction is to create
    a "blank class" whose only function is to provide _cp_dispatch.

    If there are path elements after the arguments, or more arguments
    are requested than are available in the vpath, then the 'handler'
    keyword argument specifies the next object to handle the parameterized
    request.  If handler is not specified or is None, then self is used.
    If handler is a function rather than an instance, then that function
    will be called with the args specified and the return value from that
    function used as the next object INSTEAD of adding the parameters to
    cherrypy.request.args.

    This decorator may be used in one of two ways:

    As a class decorator:

    .. code-block:: python

        @cherrypy.popargs('year', 'month', 'day')
        class Blog:
            def index(self, year=None, month=None, day=None):
                #Process the parameters here; any url like
                #/, /2009, /2009/12, or /2009/12/31
                #will fill in the appropriate parameters.

            def create(self):
                #This link will still be available at /create.
                #Defined functions take precedence over arguments.

    Or as a member of a class:

    .. code-block:: python

        class Blog:
            _cp_dispatch = cherrypy.popargs('year', 'month', 'day')
            #...

    The handler argument may be used to mix arguments with built in functions.
    For instance, the following setup allows different activities at the
    day, month, and year level:

    .. code-block:: python

        class DayHandler:
            def index(self, year, month, day):
                #Do something with this day; probably list entries

            def delete(self, year, month, day):
                #Delete all entries for this day

        @cherrypy.popargs('day', handler=DayHandler())
        class MonthHandler:
            def index(self, year, month):
                #Do something with this month; probably list entries

            def delete(self, year, month):
                #Delete all entries for this month

        @cherrypy.popargs('month', handler=MonthHandler())
        class YearHandler:
            def index(self, year):
                #Do something with this year

            #...

        @cherrypy.popargs('year', handler=YearHandler())
        class Root:
            def index(self):
                #...

    """
    handler = None
    handler_call = False
    for k, v in kwargs.items():
        if k == 'handler':
            handler = v
        else:
            tm = "cherrypy.popargs() got an unexpected keyword argument '{0}'"
            raise TypeError(tm.format(k))
    import inspect
    if handler is not None and (hasattr(handler, '__call__') or inspect.isclass(handler)):
        handler_call = True

    def decorated(cls_or_self=None, vpath=None):
        if inspect.isclass(cls_or_self):
            cls = cls_or_self
            name = cherrypy.dispatch.Dispatcher.dispatch_method_name
            setattr(cls, name, decorated)
            return cls
        self = cls_or_self
        parms = {}
        for arg in args:
            if not vpath:
                break
            parms[arg] = vpath.pop(0)
        if handler is not None:
            if handler_call:
                return handler(**parms)
            else:
                cherrypy.request.params.update(parms)
                return handler
        cherrypy.request.params.update(parms)
        if vpath:
            return getattr(self, vpath.pop(0), None)
        else:
            return self
    return decorated