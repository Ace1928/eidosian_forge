from __future__ import absolute_import, division, print_function
import datetime
import uuid
def is_dict_item_present_on_server(server_list_of_dict, dict_to_check):
    if dict_to_check is None and server_list_of_dict is None:
        return True
    if len(dict_to_check) == 0:
        return False
    if type(server_list_of_dict) is not list:
        return False
    for server_dict in server_list_of_dict:
        if (dict_to_check.items() <= server_dict.items()) is True:
            return True
    return False