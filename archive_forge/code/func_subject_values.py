from . import schema
from .jsonutil import get_column
from .search import Search
def subject_values(self, project=None):
    """ Look for the values a the subject level in the database.

            .. note::
                Is equivalent to interface.select('//subjects').get()
        """
    self._intf._get_entry_point()
    uri = '%s/subjects?columns=ID' % self._intf._entry
    if project is not None:
        uri += '&project=%s' % project
    return get_column(self._get_json(uri), 'ID')