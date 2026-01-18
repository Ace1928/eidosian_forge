from __future__ import (absolute_import, division, print_function)
import re
import platform
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def ss_parse(raw):
    """
    The ss_parse result can be either split in 6 or 7 elements depending on the process column,
    e.g. due to unprivileged user.
    :param raw: ss raw output String. First line explains the format, each following line contains a connection.
    :return: List of dicts, each dict contains protocol, state, local address, foreign address, port, name, pid for one
     connection.
    """
    results = list()
    regex_conns = re.compile(pattern='\\[?(.+?)\\]?:([0-9]+)$')
    regex_pid = re.compile(pattern='"(.*?)",pid=(\\d+)')
    lines = raw.splitlines()
    if len(lines) == 0 or not lines[0].startswith('Netid '):
        raise EnvironmentError('Unknown stdout format of `ss`: {0}'.format(raw))
    lines = lines[1:]
    for line in lines:
        cells = line.split(None, 6)
        try:
            if len(cells) == 6:
                process = str()
                protocol, state, recv_q, send_q, local_addr_port, peer_addr_port = cells
            else:
                protocol, state, recv_q, send_q, local_addr_port, peer_addr_port, process = cells
        except ValueError:
            raise EnvironmentError('Expected `ss` table layout "Netid, State, Recv-Q, Send-Q, Local Address:Port, Peer Address:Port" and                  optionally "Process", but got something else: {0}'.format(line))
        conns = regex_conns.search(local_addr_port)
        pids = regex_pid.findall(process)
        if conns is None and pids is None:
            continue
        if pids is None:
            pids = [(str(), 0)]
        address = conns.group(1)
        port = conns.group(2)
        for name, pid in pids:
            result = {'protocol': protocol, 'state': state, 'address': address, 'foreign_address': peer_addr_port, 'port': int(port), 'name': name, 'pid': int(pid)}
            results.append(result)
    return results