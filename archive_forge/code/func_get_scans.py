from lxml import etree
import urllib
from .search import SearchManager
from .users import Users
from .resources import Project
from .tags import Tags
from .jsonutil import JsonTable
def get_scans(self, triple):
    return JsonTable(self._intf._get_json('/data/prearchive/projects/%s/scans' % '/'.join(triple))).get('ID')