from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_map_config(self):
    if not self.dot_data:
        self._module.fail_json(msg="'data' is mandatory with state 'present'")
    graph = self._build_graph()
    nodes = self._get_graph_nodes(graph)
    edges = self._get_graph_edges(graph)
    icon_ids = self._get_icon_ids()
    map_config = {'name': self.map_name, 'label_type': self._get_label_type_id(self.label_type), 'expandproblem': int(self.expand_problem), 'highlight': int(self.highlight), 'width': self.width, 'height': self.height, 'selements': self._get_selements(graph, nodes, icon_ids), 'links': self._get_links(nodes, edges)}
    return map_config