from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netvisor.pn_nvos import pn_cli, run_cli
from ansible_collections.community.network.plugins.module_utils.network.netvisor.netvisor import run_commands
def check_cli(module, cli):
    """
    This method checks for pim ssm config using the vrouter-show command.
    If a user already exists on the given switch, return True else False.
    :param module: The Ansible module to fetch input parameters
    :param cli: The CLI string
    """
    name = module.params['pn_vrouter_name']
    network = module.params['pn_network']
    show = cli
    cli += ' vrouter-show name %s format name no-show-headers' % name
    rc, out, err = run_commands(module, cli)
    VROUTER_EXISTS = '' if out else None
    cli = show
    cli += ' vrouter-bgp-network-show vrouter-name %s network %s format network no-show-headers' % (name, network)
    out = run_commands(module, cli)[1]
    out = out.split()
    NETWORK_EXISTS = True if network in out[-1] else False
    return (NETWORK_EXISTS, VROUTER_EXISTS)