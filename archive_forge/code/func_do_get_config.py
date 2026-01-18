import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_get_config(self, line):
    """get_config <peer> <source>
        eg. get_config sw1 startup
        """

    def f(p, args):
        try:
            source = args[0]
        except:
            print('argument error')
            return
        print(p.get_config(source))
    self._request(line, f)