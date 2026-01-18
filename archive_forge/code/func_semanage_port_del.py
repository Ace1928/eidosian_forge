from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_port_del(module, ports, proto, setype, do_reload, sestore='', local=False):
    """ Delete SELinux port type definition from the policy.

    :type module: AnsibleModule
    :param module: Ansible module

    :type ports: list
    :param ports: List of ports and port ranges to delete (e.g. ["8080", "8080-9090"])

    :type proto: str
    :param proto: Protocol ('tcp' or 'udp')

    :type setype: str
    :param setype: SELinux type.

    :type do_reload: bool
    :param do_reload: Whether to reload SELinux policy after commit

    :type sestore: str
    :param sestore: SELinux store

    :rtype: bool
    :return: True if the policy was changed, otherwise False
    """
    change = False
    try:
        seport = seobject.portRecords(sestore)
        seport.set_reload(do_reload)
        ports_by_type = semanage_port_get_ports(seport, setype, proto, local)
        for port in ports:
            if port in ports_by_type:
                change = True
                if not module.check_mode:
                    seport.delete(port, proto)
    except (ValueError, IOError, KeyError, OSError, RuntimeError) as e:
        module.fail_json(msg='%s: %s\n' % (e.__class__.__name__, to_native(e)), exception=traceback.format_exc())
    return change