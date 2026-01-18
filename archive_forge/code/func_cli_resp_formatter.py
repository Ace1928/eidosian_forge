import gc
import sys
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
@classmethod
def cli_resp_formatter(cls, resp):
    if resp.status == STATUS_ERROR:
        return Command.cli_resp_formatter(resp)
    val = resp.value
    ret = 'Unreachable objects: {0}\n'.format(val.get('unreachable', None))
    ret += 'Total memory used (MB): {0}\n'.format(val.get('total', None))
    ret += 'Classes with instances that take-up more than one MB:\n'
    ret += '{0:<20s} {1:>16s} {2:>16s}\n'.format('Class', '#Instance', 'Size(MB)')
    for s in val.get('summary', []):
        ret += '{0:<20s} {1:>16d} {2:>16d}\n'.format(s.get('class', None), s.get('instances', None), s.get('size', None))
    return ret