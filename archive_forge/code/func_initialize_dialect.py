from __future__ import absolute_import, division, print_function
import csv
from io import BytesIO, StringIO
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
def initialize_dialect(dialect, **kwargs):

    class unix_dialect(csv.Dialect):
        """Describe the usual properties of Unix-generated CSV files."""
        delimiter = ','
        quotechar = '"'
        doublequote = True
        skipinitialspace = False
        lineterminator = '\n'
        quoting = csv.QUOTE_ALL
    csv.register_dialect('unix', unix_dialect)
    if dialect not in csv.list_dialects():
        raise DialectNotAvailableError("Dialect '%s' is not supported by your version of python." % dialect)
    dialect_params = dict(((k, v) for k, v in kwargs.items() if v is not None))
    if dialect_params:
        try:
            csv.register_dialect('custom', dialect, **dialect_params)
        except TypeError as e:
            raise CustomDialectFailureError('Unable to create custom dialect: %s' % to_native(e))
        dialect = 'custom'
    return dialect