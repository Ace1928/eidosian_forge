from __future__ import (absolute_import, division, print_function)
import sys
import time
import socket
import array
import json
import time
import re
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict
def bgpNeighborAFConfig(module, cmd, prompt, answer):
    retVal = ''
    command = ''
    bgpNeighborAFArg1 = module.params['bgpArg6']
    bgpNeighborAFArg2 = module.params['bgpArg7']
    bgpNeighborAFArg3 = module.params['bgpArg8']
    deviceType = module.params['deviceType']
    if bgpNeighborAFArg1 == 'allowas-in':
        command = command + bgpNeighborAFArg1 + ' '
        if bgpNeighborAFArg2 is not None:
            value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_af_occurances', bgpNeighborAFArg2)
            if value == 'ok':
                command = command + bgpNeighborAFArg2
            else:
                retVal = 'Error-325'
                return retVal
    elif bgpNeighborAFArg1 == 'default-originate':
        command = command + bgpNeighborAFArg1 + ' '
        if bgpNeighborAFArg2 is not None and bgpNeighborAFArg2 == 'route-map':
            command = command + bgpNeighborAFArg2 + ' '
            value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_af_routemap', bgpNeighborAFArg2)
            if value == 'ok':
                command = command + bgpNeighborAFArg3
            else:
                retVal = 'Error-324'
                return retVal
    elif bgpNeighborAFArg1 == 'filter-list':
        command = command + bgpNeighborAFArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_af_filtername', bgpNeighborAFArg2)
        if value == 'ok':
            command = command + bgpNeighborAFArg2 + ' '
            if bgpNeighborAFArg3 == 'in' or bgpNeighborAFArg3 == 'out':
                command = command + bgpNeighborAFArg3
            else:
                retVal = 'Error-323'
                return retVal
        else:
            retVal = 'Error-322'
            return retVal
    elif bgpNeighborAFArg1 == 'maximum-prefix':
        command = command + bgpNeighborAFArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_af_maxprefix', bgpNeighborAFArg2)
        if value == 'ok':
            command = command + bgpNeighborAFArg2 + ' '
            if bgpNeighborAFArg3 is not None:
                command = command + bgpNeighborAFArg3
            else:
                command = command.strip()
        else:
            retVal = 'Error-326'
            return retVal
    elif bgpNeighborAFArg1 == 'next-hop-self':
        command = command + bgpNeighborAFArg1
    elif bgpNeighborAFArg1 == 'prefix-list':
        command = command + bgpNeighborAFArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_af_prefixname', bgpNeighborAFArg2)
        if value == 'ok':
            command = command + bgpNeighborAFArg2 + ' '
            if bgpNeighborAFArg3 == 'in' or bgpNeighborAFArg3 == 'out':
                command = command + bgpNeighborAFArg3
            else:
                retVal = 'Error-321'
                return retVal
        else:
            retVal = 'Error-320'
            return retVal
    elif bgpNeighborAFArg1 == 'route-map':
        command = command + bgpNeighborAFArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_af_routemap', bgpNeighborAFArg2)
        if value == 'ok':
            command = command + bgpNeighborAFArg2
        else:
            retVal = 'Error-319'
            return retVal
    elif bgpNeighborAFArg1 == 'route-reflector-client':
        command = command + bgpNeighborAFArg1
    elif bgpNeighborAFArg1 == 'send-community':
        command = command + bgpNeighborAFArg1 + ' '
        if bgpNeighborAFArg2 is not None and bgpNeighborAFArg2 == 'extended':
            command = command + bgpNeighborAFArg2
    elif bgpNeighborAFArg1 == 'soft-reconfiguration':
        command = command + bgpNeighborAFArg1 + ' inbound'
    elif bgpNeighborAFArg1 == 'unsuppress-map':
        command = command + bgpNeighborAFArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_af_routemap', bgpNeighborAFArg2)
        if value == 'ok':
            command = command + bgpNeighborAFArg2
        else:
            retVal = 'Error-318'
            return retVal
    else:
        retVal = 'Error-317'
        return retVal
    inner_cmd = [{'command': command, 'prompt': None, 'answer': None}]
    cmd.extend(inner_cmd)
    retVal = retVal + str(cnos.run_cnos_commands(module, cmd))
    return retVal