from xml.parsers.expat import ParserCreate
from dbus.exceptions import IntrospectionParserException
def process_introspection_data(data):
    """Return a dict mapping ``interface.method`` strings to the
    concatenation of all their 'in' parameters, and mapping
    ``interface.signal`` strings to the concatenation of all their
    parameters.

    Example output::

        {
            'com.example.SignalEmitter.OneString': 's',
            'com.example.MethodImplementor.OneInt32Argument': 'i',
        }

    :Parameters:
        `data` : str
            The introspection XML. Must be an 8-bit string of UTF-8.
    """
    try:
        return _Parser().parse(data)
    except Exception as e:
        raise IntrospectionParserException('%s: %s' % (e.__class__, e))