import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def saved_templates(self, with_description=False):
    """ Returns the names of accessible saved search templates on the server.
        """
    self._intf._get_entry_point()
    jdata = self._intf._get_json('%s/search/saved?format=json' % self._intf._entry)
    if with_description:
        return [(ld['brief_description'].split('template_')[1], ld['description'].replace('%%', '%')) for ld in get_selection(jdata, ['brief_description', 'description']) if ld['brief_description'].startswith('template_')]
    else:
        return [name.split('template_')[1] for name in get_column(jdata, 'brief_description') if name.startswith('template_')]