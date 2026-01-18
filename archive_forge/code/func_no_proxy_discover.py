from __future__ import absolute_import, division, print_function
import json
import multiprocessing
import threading
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import request
from ansible.module_utils._text import to_native
def no_proxy_discover(self):
    """Discover E-Series storage systems using embedded web services."""
    thread_pool_size = min(multiprocessing.cpu_count() * self.CPU_THREAD_MULTIPLE, self.MAX_THREAD_POOL_SIZE)
    subnet = list(ipaddress.ip_network(u'%s' % self.subnet_mask))
    thread_pool = []
    search_count = len(subnet)
    for start in range(0, search_count, thread_pool_size):
        end = search_count if search_count - start < thread_pool_size else start + thread_pool_size
        for address in subnet[start:end]:
            thread = threading.Thread(target=self.check_ip_address, args=(self.systems_found, address))
            thread_pool.append(thread)
            thread.start()
        for thread in thread_pool:
            thread.join()