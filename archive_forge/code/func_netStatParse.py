from __future__ import (absolute_import, division, print_function)
import re
import platform
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def netStatParse(raw):
    """
    The netstat result can be either split in 6,7 or 8 elements depending on the values of state, process and name.
    For UDP the state is always empty. For UDP and TCP the process can be empty.
    So these cases have to be checked.
    :param raw: Netstat raw output String. First line explains the format, each following line contains a connection.
    :return: List of dicts, each dict contains protocol, state, local address, foreign address, port, name, pid for one
     connection.
    """
    results = list()
    for line in raw.splitlines():
        if line.startswith(('tcp', 'udp')):
            state = ''
            pid_and_name = ''
            process = ''
            formatted_line = line.split()
            protocol, recv_q, send_q, address, foreign_address, rest = (formatted_line[0], formatted_line[1], formatted_line[2], formatted_line[3], formatted_line[4], formatted_line[5:])
            address, port = address.rsplit(':', 1)
            if protocol.startswith('tcp'):
                protocol = 'tcp'
                if len(rest) == 3:
                    state, pid_and_name, process = rest
                if len(rest) == 2:
                    state, pid_and_name = rest
            if protocol.startswith('udp'):
                protocol = 'udp'
                if len(rest) == 2:
                    pid_and_name, process = rest
                if len(rest) == 1:
                    pid_and_name = rest[0]
            pid, name = split_pid_name(pid_name=pid_and_name)
            result = {'protocol': protocol, 'state': state, 'address': address, 'foreign_address': foreign_address, 'port': int(port), 'name': name, 'pid': int(pid)}
            if result not in results:
                results.append(result)
    return results