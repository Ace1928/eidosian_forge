import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_list_queue(self, line):
    """list_queue <peer>
        """

    def f(p, args):
        o = p.get()
        if o.resources.queue:
            for q in o.resources.queue:
                print('%s %s' % (q.resource_id, q.port))
    self._request(line, f)