import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def build_criteria_set(container_node, criteria_set):
    for criteria in criteria_set:
        if isinstance(criteria, str):
            container_node.set('method', criteria)
        if isinstance(criteria, list):
            sub_container_node = etree.Element(etree.QName(search_nsmap['xdat'], 'child_set'), nsmap=search_nsmap)
            container_node.append(build_criteria_set(sub_container_node, criteria))
        if isinstance(criteria, tuple):
            if len(criteria) != 3:
                raise ProgrammingError('%s should be a 3-element tuple' % str(criteria))
            constraint_node = etree.Element(etree.QName(search_nsmap['xdat'], 'criteria'), nsmap=search_nsmap)
            constraint_node.set('override_value_formatting', '0')
            schema_field_node = etree.Element(etree.QName(search_nsmap['xdat'], 'schema_field'), nsmap=search_nsmap)
            schema_field_node.text = criteria[0]
            comparison_type_node = etree.Element(etree.QName(search_nsmap['xdat'], 'comparison_type'), nsmap=search_nsmap)
            comparison_type_node.text = special_ops.get(criteria[1], criteria[1])
            value_node = etree.Element(etree.QName(search_nsmap['xdat'], 'value'), nsmap=search_nsmap)
            value_node.text = criteria[2].replace('*', special_ops['*'])
            constraint_node.extend([schema_field_node, comparison_type_node, value_node])
            container_node.append(constraint_node)
    return container_node