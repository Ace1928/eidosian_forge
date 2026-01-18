def simplefilter(f):
    """
    Decorator that converts a function into a filter::

        @simplefilter
        def lowercase(self, lexer, stream, options):
            for ttype, value in stream:
                yield ttype, value.lower()
    """
    return type(f.__name__, (FunctionFilter,), {'__module__': getattr(f, '__module__'), '__doc__': f.__doc__, 'function': f})