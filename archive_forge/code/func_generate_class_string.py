import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def generate_class_string(name, props, project_shortname, prefix):
    package_name = snake_case_to_camel_case(project_shortname)
    props = reorder_props(props=props)
    prop_keys = list(props.keys())
    wildcards = ''
    wildcard_declaration = ''
    wildcard_names = ''
    default_paramtext = ''
    default_argtext = ''
    accepted_wildcards = ''
    if any((key.endswith('-*') for key in prop_keys)):
        accepted_wildcards = get_wildcards_r(prop_keys)
        wildcards = ', ...'
        wildcard_declaration = wildcard_template.format(accepted_wildcards.replace('-*', ''))
        wildcard_names = ', wildcard_names'
    prop_names = ', '.join(("'{}'".format(p) for p in prop_keys if '*' not in p and p not in ['setProps']))
    for item in prop_keys[:]:
        if item.endswith('-*') or item == 'setProps':
            prop_keys.remove(item)
        elif item in r_keywords:
            prop_keys.remove(item)
            warnings.warn('WARNING: prop "{}" in component "{}" is an R keyword - REMOVED FROM THE R COMPONENT'.format(item, name))
    default_argtext += ', '.join(('{}=NULL'.format(p) for p in prop_keys))
    default_paramtext += ', '.join(('{0}={0}'.format(p) if p != 'children' else '{}=children'.format(p) for p in prop_keys))
    return r_component_string.format(funcname=format_fn_name(prefix, name), name=name, default_argtext=default_argtext, wildcards=wildcards, wildcard_declaration=wildcard_declaration, default_paramtext=default_paramtext, project_shortname=project_shortname, prop_names=prop_names, wildcard_names=wildcard_names, package_name=package_name)