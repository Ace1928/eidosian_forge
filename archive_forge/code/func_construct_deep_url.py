from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def construct_deep_url(self, target_object, parent_objects=None, child_classes=None):
    """
        This method is used to retrieve the appropriate URL path and filter_string to make the request to the APIC.

        :param target_object: The target class dictionary containing parent_class, aci_class, aci_rn, target_filter, and module_object keys.
        :param parent_objects: The parent class list of dictionaries containing parent_class, aci_class, aci_rn, target_filter, and module_object keys.
        :param child_classes: The list of child classes that the module supports along with the object.
        :type target_object: dict
        :type parent_objects: list[dict]
        :type child_classes: list[string]
        :return: The path and filter_string needed to build the full URL.
        """
    self.filter_string = ''
    rn_builder = None
    subtree_classes = None
    add_subtree_filter = False
    add_target_filter = False
    has_target_query = False
    has_target_query_compare = False
    has_target_query_difference = False
    has_target_query_called = False
    if child_classes is None:
        self.child_classes = set()
    else:
        self.child_classes = set(child_classes)
    target_parent_class = target_object.get('parent_class')
    target_class = target_object.get('aci_class')
    target_rn = target_object.get('aci_rn')
    target_filter = target_object.get('target_filter')
    target_module_object = target_object.get('module_object')
    url_path_object = dict(target_class=target_class, target_filter=target_filter, subtree_class=target_class, subtree_filter=target_filter, module_object=target_module_object)
    if target_module_object is not None:
        rn_builder = target_rn
    else:
        has_target_query = True
        has_target_query_compare = True
    if parent_objects is not None:
        current_parent_class = target_parent_class
        has_parent_query_compare = False
        has_parent_query_difference = False
        is_first_parent = True
        is_single_parent = None
        search_classes = set()
        while current_parent_class != 'uni':
            parent_object = self._deep_url_parent_object(parent_objects=parent_objects, parent_class=current_parent_class)
            if parent_object is not None:
                parent_parent_class = parent_object.get('parent_class')
                parent_class = parent_object.get('aci_class')
                parent_rn = parent_object.get('aci_rn')
                parent_filter = parent_object.get('target_filter')
                parent_module_object = parent_object.get('module_object')
                if is_first_parent:
                    is_single_parent = True
                else:
                    is_single_parent = False
                is_first_parent = False
                if parent_parent_class != 'uni':
                    search_classes.add(parent_class)
                if parent_module_object is not None:
                    if rn_builder is not None:
                        rn_builder = '{0}/{1}'.format(parent_rn, rn_builder)
                    else:
                        rn_builder = parent_rn
                    url_path_object['target_class'] = parent_class
                    url_path_object['target_filter'] = parent_filter
                    has_target_query = False
                else:
                    rn_builder = None
                    subtree_classes = search_classes
                    has_target_query = True
                    if is_single_parent:
                        has_parent_query_compare = True
                current_parent_class = parent_parent_class
            else:
                raise ValueError("Reference error for parent_class '{0}'. Each parent_class must reference a valid object".format(current_parent_class))
            if not has_target_query_difference and (not has_target_query_called):
                if has_target_query is not has_target_query_compare:
                    has_target_query_difference = True
            elif not has_parent_query_difference and has_target_query is not has_parent_query_compare:
                has_parent_query_difference = True
            has_target_query_called = True
        if not has_parent_query_difference and has_parent_query_compare and (target_module_object is not None):
            add_target_filter = True
        elif has_parent_query_difference and target_module_object is not None:
            add_subtree_filter = True
            self.child_classes.add(target_class)
            if has_target_query:
                add_target_filter = True
        elif has_parent_query_difference and (not has_target_query) and (target_module_object is None):
            self.child_classes.add(target_class)
            self.child_classes.update(subtree_classes)
        elif not has_parent_query_difference and (not has_target_query) and (target_module_object is None):
            self.child_classes.add(target_class)
        elif not has_target_query and is_single_parent and (target_module_object is None):
            self.child_classes.add(target_class)
    url_path_object['object_rn'] = rn_builder
    url_path_object['add_subtree_filter'] = add_subtree_filter
    url_path_object['add_target_filter'] = add_target_filter
    self._deep_url_path_builder(url_path_object)