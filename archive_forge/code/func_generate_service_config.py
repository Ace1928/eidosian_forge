from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def generate_service_config(self, name, sortorder, weight, algorithm, description, tags, problem_tags, parents, children, propagation_rule, propagation_value, status_rules):
    algorithms = {'status_to_ok': '0', 'most_crit_if_all_children': '1', 'most_crit_of_child_serv': '2'}
    algorithm = algorithms[algorithm]
    request = {'name': name, 'algorithm': algorithm, 'sortorder': sortorder, 'description': description, 'weight': weight}
    if tags:
        request['tags'] = tags
    else:
        request['tags'] = []
    request['problem_tags'] = []
    if problem_tags:
        p_operators = {'equals': '0', 'like': '2'}
        for p_tag in problem_tags:
            pt = {'tag': p_tag['tag'], 'operator': '0', 'value': ''}
            if 'operator' in p_tag:
                pt['operator'] = p_operators[p_tag['operator']]
            if 'value' in p_tag:
                pt['value'] = p_tag['value']
            request['problem_tags'].append(pt)
    if parents:
        p_service_ids = []
        p_services = self._zapi.service.get({'filter': {'name': parents}})
        for p_service in p_services:
            p_service_ids.append({'serviceid': p_service['serviceid']})
        request['parents'] = p_service_ids
    else:
        request['parents'] = []
    if children:
        c_service_ids = []
        c_services = self._zapi.service.get({'filter': {'name': children}})
        for c_service in c_services:
            c_service_ids.append({'serviceid': c_service['serviceid']})
        request['children'] = c_service_ids
    else:
        request['children'] = []
    request['status_rules'] = []
    if status_rules:
        for s_rule in status_rules:
            status_rule = {}
            if 'type' in s_rule:
                sr_type_map = {'at_least_n_child_services_have_status_or_above': '0', 'at_least_npct_child_services_have_status_or_above': '1', 'less_than_n_child_services_have_status_or_below': '2', 'less_than_npct_child_services_have_status_or_below': '3', 'weight_child_services_with_status_or_above_at_least_w': '4', 'weight_child_services_with_status_or_above_at_least_npct': '5', 'weight_child_services_with_status_or_below_less_w': '6', 'weight_child_services_with_status_or_below_less_npct': '7'}
                if s_rule['type'] not in sr_type_map:
                    self._module.fail_json(msg="Wrong value for 'type' parameter in status rule.")
                status_rule['type'] = sr_type_map[s_rule['type']]
            else:
                self._module.fail_json(msg="'type' is mandatory paremeter for status rule.")
            if 'limit_value' in s_rule:
                lv = s_rule['limit_value']
                if status_rule['type'] in ['0', '2', '4', '6']:
                    if int(lv) < 1 or int(lv) > 100000:
                        self._module.fail_json(msg="'limit_value' for N and W must be between 1 and 100000 but provided %s" % lv)
                elif int(lv) < 1 or int(lv) > 100:
                    self._module.fail_json(msg="'limit_value' for N%% must be between 1 and 100 but provided %s" % lv)
                status_rule['limit_value'] = str(lv)
            else:
                self._module.fail_json(msg="'limit_value' is mandatory paremeter for status rule.")
            if 'limit_status' in s_rule:
                sr_ls_map = {'ok': '-1', 'not_classified': '0', 'information': '1', 'warning': '2', 'average': '3', 'high': '4', 'disaster': 5}
                if s_rule['limit_status'] not in sr_ls_map:
                    self._module.fail_json(msg="Wrong value for 'limit_status' parameter in status rule.")
                status_rule['limit_status'] = sr_ls_map[s_rule['limit_status']]
            else:
                self._module.fail_json(msg="'limit_status' is mandatory paremeter for status rule.")
            if 'new_status' in s_rule:
                sr_ns_map = {'not_classified': '0', 'information': '1', 'warning': '2', 'average': '3', 'high': '4', 'disaster': '5'}
                if s_rule['new_status'] not in sr_ns_map:
                    self._module.fail_json(msg="Wrong value for 'new_status' parameter in status rule.")
                status_rule['new_status'] = sr_ns_map[s_rule['new_status']]
            else:
                self._module.fail_json(msg="'new_status' is mandatory paremeter for status rule.")
            request['status_rules'].append(status_rule)
    request['propagation_rule'] = '0'
    if propagation_rule:
        if propagation_value is None:
            self._module.fail_json(msg="If 'propagation_rule' is provided then 'propagation_value' must be provided too.")
        pr_map = {'as_is': '0', 'increase': '1', 'decrease': '2', 'ignore': '3', 'fixed': '4'}
        if propagation_rule not in pr_map:
            self._module.fail_json(msg="Wrong value for 'propagation_rule' parameter.")
        else:
            request['propagation_rule'] = pr_map[propagation_rule]
    request['propagation_value'] = '0'
    if propagation_value:
        if propagation_rule is None:
            self._module.fail_json(msg="If 'propagation_value' is provided then 'propagation_rule' must be provided too.")
        pv_map = {'ok': '-1', 'not_classified': '0', 'information': '1', 'warning': '2', 'average': '3', 'high': '4', 'disaster': '5'}
        if propagation_value not in pv_map:
            self._module.fail_json(msg="Wrong value for 'propagation_value' parameter.")
        else:
            request['propagation_value'] = pv_map[propagation_value]
    return request