import re
import sys
import types
import fnmatch
from os.path import basename
from pygments.formatters._mapping import FORMATTERS
from pygments.plugin import find_plugin_formatters
from pygments.util import ClassNotFound, itervalues
def load_formatter_from_file(filename, formattername='CustomFormatter', **options):
    """Load a formatter from a file.

    This method expects a file located relative to the current working
    directory, which contains a class named CustomFormatter. By default,
    it expects the Formatter to be named CustomFormatter; you can specify
    your own class name as the second argument to this function.

    Users should be very careful with the input, because this method
    is equivalent to running eval on the input file.

    Raises ClassNotFound if there are any problems importing the Formatter.

    .. versionadded:: 2.2
    """
    try:
        custom_namespace = {}
        exec(open(filename, 'rb').read(), custom_namespace)
        if formattername not in custom_namespace:
            raise ClassNotFound('no valid %s class found in %s' % (formattername, filename))
        formatter_class = custom_namespace[formattername]
        return formatter_class(**options)
    except IOError as err:
        raise ClassNotFound('cannot read %s' % filename)
    except ClassNotFound as err:
        raise
    except Exception as err:
        raise ClassNotFound('error when loading custom formatter: %s' % err)