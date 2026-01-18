import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def query_from_criteria_set(criteria_set):
    query = []
    query.append(criteria_set.get('method'))
    _nsmap = criteria_set.nsmap
    for criteria in criteria_set.xpath('xdat:criteria', namespaces=_nsmap):
        _f = criteria.xpath('xdat:schema_field', namespaces=_nsmap)[0]
        _o = criteria.xpath('xdat:comparison_type', namespaces=_nsmap)[0]
        _v = criteria.xpath('xdat:value', namespaces=_nsmap)[0]
        constraint = (_f.text, _o.text, _v.text)
        query.insert(0, constraint)
    for child_set in criteria_set.xpath('xdat:child_set', namespaces=_nsmap):
        query.insert(0, query_from_criteria_set(child_set))
    return query