import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def get_r_prop_types(type_object):
    """Mapping from the PropTypes js type object to the R type."""

    def shape_or_exact():
        return 'lists containing elements {}.\n{}'.format(', '.join(("'{}'".format(t) for t in type_object['value'])), 'Those elements have the following types:\n{}'.format('\n'.join((create_prop_docstring_r(prop_name=prop_name, type_object=prop, required=prop['required'], description=prop.get('description', ''), indent_num=1) for prop_name, prop in type_object['value'].items()))))
    return dict(array=lambda: 'unnamed list', bool=lambda: 'logical', number=lambda: 'numeric', string=lambda: 'character', object=lambda: 'named list', any=lambda: 'logical | numeric | character | named list | unnamed list', element=lambda: 'dash component', node=lambda: 'a list of or a singular dash component, string or number', enum=lambda: 'a value equal to: {}'.format(', '.join(('{}'.format(str(t['value'])) for t in type_object['value']))), union=lambda: '{}'.format(' | '.join(('{}'.format(get_r_type(subType)) for subType in type_object['value'] if get_r_type(subType) != ''))), arrayOf=lambda: 'list' + (' of {}s'.format(get_r_type(type_object['value'])) if get_r_type(type_object['value']) != '' else ''), objectOf=lambda: 'list with named elements and values of type {}'.format(get_r_type(type_object['value'])), shape=shape_or_exact, exact=shape_or_exact)