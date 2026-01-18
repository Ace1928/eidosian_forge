import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_get_queue_config(self, line):
    """get_queue_port <peer> <source> <queue>
        eg. get_queue_config sw1 running LogicalSwitch7-Port1-Queue922
        """

    def f(p, args):
        try:
            source, queue = args
        except:
            print('argument error')
            return
        o = p.get_config(source)
        for q in o.resources.queue:
            if q.resource_id != queue:
                continue
            print(q.resource_id)
            conf = q.properties
            for k in self._queue_settings:
                try:
                    v = getattr(conf, k)
                except AttributeError:
                    continue
                print('%s %s' % (k, v))
    self._request(line, f)