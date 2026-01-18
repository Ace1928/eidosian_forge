import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def query_from_xml(document):
    query = {}
    root = etree.fromstring(document)
    _nsmap = root.nsmap
    query['description'] = root.get('description', default='')
    query['row'] = root.xpath('xdat:root_element_name', namespaces=root.nsmap)[0].text
    query['columns'] = []
    for node in root.xpath('xdat:search_field', namespaces=_nsmap):
        en = node.xpath('xdat:element_name', namespaces=root.nsmap)[0].text
        fid = node.xpath('xdat:field_ID', namespaces=root.nsmap)[0].text
        query['columns'].append('%s/%s' % (en, fid))
    query['users'] = [node.text for node in root.xpath('xdat:allowed_user/xdat:login', namespaces=root.nsmap)]
    try:
        search_where = root.xpath('xdat:search_where', namespaces=root.nsmap)[0]
        query['constraints'] = query_from_criteria_set(search_where)
    except Exception:
        query['constraints'] = [('%s/ID' % query['row'], 'LIKE', '%'), 'AND']
    return query