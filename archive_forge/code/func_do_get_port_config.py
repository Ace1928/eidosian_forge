import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_get_port_config(self, line):
    """get_config_port <peer> <source> <port>
        eg. get_port_config sw1 running LogicalSwitch7-Port2
        """

    def f(p, args):
        try:
            source, port = args
        except:
            print('argument error')
            return
        o = p.get_config(source)
        for p in o.resources.port:
            if p.resource_id != port:
                continue
            print(p.resource_id)
            conf = p.configuration
            for k in self._port_settings:
                try:
                    v = getattr(conf, k)
                except AttributeError:
                    continue
                print('%s %s' % (k, v))
    self._request(line, f)