from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.namespace import PrefixFactNamespace
from ansible.module_utils.facts import default_collectors
from ansible.module_utils.facts import ansible_collector
def ansible_facts(module, gather_subset=None):
    """Compat api for ansible 2.0/2.2/2.3 module_utils.facts.ansible_facts method

    2.3/2.3 expects a gather_subset arg.
    2.0/2.1 does not except a gather_subset arg

    So make gather_subsets an optional arg, defaulting to configured DEFAULT_GATHER_TIMEOUT

    'module' should be an instance of an AnsibleModule.

    returns a dict mapping the bare fact name ('default_ipv4' with no 'ansible_' namespace) to
    the fact value.
    """
    gather_subset = gather_subset or module.params.get('gather_subset', ['all'])
    gather_timeout = module.params.get('gather_timeout', 10)
    filter_spec = module.params.get('filter', '*')
    minimal_gather_subset = frozenset(['apparmor', 'caps', 'cmdline', 'date_time', 'distribution', 'dns', 'env', 'fips', 'local', 'lsb', 'pkg_mgr', 'platform', 'python', 'selinux', 'service_mgr', 'ssh_pub_keys', 'user'])
    all_collector_classes = default_collectors.collectors
    namespace = PrefixFactNamespace(namespace_name='ansible', prefix='')
    fact_collector = ansible_collector.get_ansible_collector(all_collector_classes=all_collector_classes, namespace=namespace, filter_spec=filter_spec, gather_subset=gather_subset, gather_timeout=gather_timeout, minimal_gather_subset=minimal_gather_subset)
    facts_dict = fact_collector.collect(module=module)
    return facts_dict