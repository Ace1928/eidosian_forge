import cmd
import sys
import lxml.etree as ET
from ncclient.operations.rpc import RPCError
from os_ken import cfg
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
import os_ken.lib.of_config.classes as ofc
def do_show_logical_switch(self, line):
    """show_logical_switch <peer> <logical switch>
        """

    def f(p, args):
        try:
            lsw, = args
        except:
            print('argument error')
            return
        o = p.get()
        for s in o.logical_switches.switch:
            if s.id != lsw:
                continue
            print(s.id)
            print('datapath-id %s' % s.datapath_id)
            if s.resources.queue:
                print('queues:')
                for q in s.resources.queue:
                    print('\t %s' % q)
            if s.resources.port:
                print('ports:')
                for p in s.resources.port:
                    print('\t %s' % p)
    self._request(line, f)