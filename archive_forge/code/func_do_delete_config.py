import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_delete_config(self, line):
    """delete_config <peer> <source>
        eg. delete_config sw1 startup
        """

    def f(p, args):
        try:
            source = args[0]
        except:
            print('argument error')
            return
        print(p.delete_config(source))
    self._request(line, f)