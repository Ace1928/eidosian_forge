from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def derive_config_from_deleted_cmd_dict(command, exist_conf, test_keys=None, key_set=None, key_match_op=None, delete_op=None):
    if test_keys is None:
        test_keys = []
    if key_set is None:
        key_set = set()
    if key_match_op is None:
        key_match_op = __KEY_MATCH_OP_DEFAULT
    if delete_op is None:
        delete_op = __DELETE_OP_DEFAULT
    new_conf = deepcopy(exist_conf)
    if not command:
        return (True, [])
    trival_cmd_key_set, dict_list_cmd_key_set = get_key_sets(command)
    trival_exist_key_set, dict_list_exist_key_set = get_key_sets(new_conf)
    common_trival_key_set = trival_cmd_key_set.intersection(trival_exist_key_set)
    common_dict_list_key_set = dict_list_cmd_key_set.intersection(dict_list_exist_key_set)
    key_matched = key_match_op(key_set, command, new_conf)
    if key_matched:
        done, new_conf = delete_op(key_set, command, new_conf)
        if done:
            return (key_matched, new_conf)
        else:
            nu, dict_list_exist_key_set = get_key_sets(new_conf)
            common_dict_list_key_set = dict_list_cmd_key_set.intersection(dict_list_exist_key_set)
    else:
        return (key_matched, new_conf)
    for key in key_set:
        common_dict_list_key_set.discard(key)
    for key in common_dict_list_key_set:
        cmd_value = command[key]
        exist_value = new_conf[key]
        t_key_set = get_test_key_set(key, test_keys)
        t_key_match_op = get_key_match_op(key, test_keys)
        t_delete_op = get_delete_op(key, test_keys)
        if isinstance(cmd_value, list) and isinstance(exist_value, list):
            c_list = cmd_value
            e_list = exist_value
            new_conf_list = list()
            not_dict_item = False
            dict_no_key_item = False
            for c_item in c_list:
                for e_item in e_list:
                    if isinstance(c_item, dict) and isinstance(e_item, dict):
                        if t_key_set:
                            remaining_keys = [t_key_item for t_key_item in test_keys if key not in t_key_item]
                            k_mtchd, new_conf_dict = derive_config_from_deleted_cmd_dict(c_item, e_item, remaining_keys, t_key_set, t_key_match_op, t_delete_op)
                            if k_mtchd:
                                new_conf[key].remove(e_item)
                                if new_conf_dict:
                                    new_conf_list.append(new_conf_dict)
                                break
                        else:
                            dict_no_key_item = True
                            break
                    else:
                        not_dict_item = True
                        break
                if not_dict_item or dict_no_key_item:
                    break
            if dict_no_key_item:
                new_conf_list = e_list
            if not_dict_item:
                c_set = set(c_list)
                e_set = set(e_list)
                delete_set = e_set.difference(c_set)
                if delete_set:
                    new_conf[key] = list(delete_set)
                else:
                    new_conf[key] = []
            elif new_conf_list:
                new_conf[key].extend(new_conf_list)
        elif isinstance(cmd_value, dict) and isinstance(exist_value, dict):
            k_mtchd, new_conf_dict = derive_config_from_deleted_cmd_dict(cmd_value, exist_value, test_keys, None, t_key_match_op, t_delete_op)
            if k_mtchd:
                new_conf.pop(key, None)
                if new_conf_dict:
                    new_conf[key] = new_conf_dict
        elif isinstance(cmd_value, (list, dict)) or isinstance(exist_value, (list, dict)):
            new_conf[key] = exist_value
            break
        else:
            continue
    return (key_matched, new_conf)