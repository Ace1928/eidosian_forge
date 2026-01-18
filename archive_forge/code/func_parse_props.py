import copy
import json
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, FeatureNotFound
from bs4.element import Tag
from . import (
from .dom_helpers import get_attr, get_children, get_descendents, try_urljoin
from .mf_helpers import unordered_list
from .version import __version__
def parse_props(el, root_lang):
    """Parse the properties from a single element"""
    props = {}
    children = []
    parsed_types_aggregation = set()
    classes = el.get('class', [])
    filtered_classes = mf2_classes.filter_classes(classes)
    root_class_names = filtered_classes['h']
    backcompat_mode = False
    if not root_class_names:
        root_class_names = backcompat.root(classes)
        backcompat_mode = True
    if root_class_names:
        parsed_types_aggregation.add('h')
    is_property_el = False
    p_value = None
    for prop_name in filtered_classes['p']:
        is_property_el = True
        parsed_types_aggregation.add('p')
        prop_value = props.setdefault(prop_name, [])
        if p_value is None:
            p_value = parse_property.text(el, base_url=self.__url__)
        if root_class_names:
            prop_value.append(handle_microformat(root_class_names, el, value_property='name', simple_value=p_value, backcompat_mode=backcompat_mode))
        else:
            prop_value.append(p_value)
    u_value = None
    for prop_name in filtered_classes['u']:
        is_property_el = True
        parsed_types_aggregation.add('u')
        prop_value = props.setdefault(prop_name, [])
        if u_value is None:
            u_value = parse_property.url(el, base_url=self.__url__)
        if root_class_names:
            prop_value.append(handle_microformat(root_class_names, el, value_property='url', simple_value=u_value, backcompat_mode=backcompat_mode))
        elif isinstance(u_value, dict):
            prop_value.append(u_value)
        else:
            prop_value.append(u_value)
    dt_value = None
    for prop_name in filtered_classes['dt']:
        is_property_el = True
        parsed_types_aggregation.add('d')
        prop_value = props.setdefault(prop_name, [])
        if dt_value is None:
            dt_value, new_date = parse_property.datetime(el, self._default_date)
            if new_date:
                self._default_date = new_date
        if root_class_names:
            stops_implied_name = True
            prop_value.append(handle_microformat(root_class_names, el, simple_value=dt_value, backcompat_mode=backcompat_mode))
        elif dt_value is not None:
            prop_value.append(dt_value)
    e_value = None
    for prop_name in filtered_classes['e']:
        is_property_el = True
        parsed_types_aggregation.add('e')
        prop_value = props.setdefault(prop_name, [])
        if e_value is None:
            if el.original is None:
                embedded_el = el
            else:
                embedded_el = el.original
            if self._preserve_doc:
                embedded_el = copy.copy(embedded_el)
            temp_fixes.rm_templates(embedded_el)
            e_value = parse_property.embedded(embedded_el, self.__url__, root_lang, self.lang, self.expose_dom)
        if root_class_names:
            stops_implied_name = True
            prop_value.append(handle_microformat(root_class_names, el, simple_value=e_value, backcompat_mode=backcompat_mode))
        else:
            prop_value.append(e_value)
    if not is_property_el and root_class_names:
        children.append(handle_microformat(root_class_names, el, backcompat_mode=backcompat_mode))
    if not root_class_names:
        for child in get_children(el):
            child_properties, child_microformats, child_parsed_types_aggregation = parse_props(child, root_lang)
            for prop_name in child_properties:
                v = props.get(prop_name, [])
                v.extend(child_properties[prop_name])
                props[prop_name] = v
            children.extend(child_microformats)
            parsed_types_aggregation.update(child_parsed_types_aggregation)
    return (props, children, parsed_types_aggregation)