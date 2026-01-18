from __future__ import (absolute_import, division, print_function)
def get_modified_attributes(self, current, desired, get_list_diff=False, additional_keys=False):
    """ takes two dicts of attributes and return a dict of attributes that are
            not in the current state
            It is expected that all attributes of interest are listed in current and
            desired.
            The same assumption holds true for any nested directory.
            TODO: This is actually not true for the ElementSW 'attributes' directory.
                  Practically it means you cannot add or remove a key in a modify.
            :param: current: current attributes in ONTAP
            :param: desired: attributes from playbook
            :param: get_list_diff: specifies whether to have a diff of desired list w.r.t current list for an attribute
            :return: dict of attributes to be modified
            :rtype: dict

            NOTE: depending on the attribute, the caller may need to do a modify or a
            different operation (eg move volume if the modified attribute is an
            aggregate name)
        """
    modified = dict()
    if current is None:
        return modified
    self.check_keys(current, desired)
    for key, value in current.items():
        if key in desired and desired[key] is not None:
            if type(value) is list:
                modified_list = self.compare_lists(value, desired[key], get_list_diff)
                if modified_list:
                    modified[key] = modified_list
            elif type(value) is dict:
                modified_dict = self.get_modified_attributes(value, desired[key], get_list_diff, additional_keys=True)
                if modified_dict:
                    modified[key] = modified_dict
            elif cmp(value, desired[key]) != 0:
                modified[key] = desired[key]
    if additional_keys:
        for key, value in desired.items():
            if key not in current:
                modified[key] = desired[key]
    if modified:
        self.changed = True
    return modified