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
def bgpNeighborConfig(module, cmd, prompt, answer):
    retVal = ''
    command = ''
    bgpNeighborArg1 = module.params['bgpArg4']
    bgpNeighborArg2 = module.params['bgpArg5']
    bgpNeighborArg3 = module.params['bgpArg6']
    bgpNeighborArg4 = module.params['bgpArg7']
    bgpNeighborArg5 = module.params['bgpArg8']
    deviceType = module.params['deviceType']
    if bgpNeighborArg1 == 'address-family':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_address_family', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2 + ' unicast'
            inner_cmd = [{'command': command, 'prompt': None, 'answer': None}]
            cmd.extend(inner_cmd)
            retVal = retVal + bgpNeighborAFConfig(module, cmd, '(config-router-neighbor-af)#', answer)
            return retVal
        else:
            retVal = 'Error-316'
            return retVal
    elif bgpNeighborArg1 == 'advertisement-interval':
        command = command + bgpNeighborArg1
    elif bgpNeighborArg1 == 'bfd':
        command = command + bgpNeighborArg1 + ' '
        if bgpNeighborArg2 is not None and bgpNeighborArg2 == 'mutihop':
            command = command + bgpNeighborArg2
    elif bgpNeighborArg1 == 'connection-retry-time':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_connection_retrytime', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2
        else:
            retVal = 'Error-315'
            return retVal
    elif bgpNeighborArg1 == 'description':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_description', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2
        else:
            retVal = 'Error-314'
            return retVal
    elif bgpNeighborArg1 == 'disallow-infinite-holdtime':
        command = command + bgpNeighborArg1
    elif bgpNeighborArg1 == 'dont-capability-negotiate':
        command = command + bgpNeighborArg1
    elif bgpNeighborArg1 == 'dynamic-capability':
        command = command + bgpNeighborArg1
    elif bgpNeighborArg1 == 'ebgp-multihop':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_maxhopcount', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2
        else:
            retVal = 'Error-313'
            return retVal
    elif bgpNeighborArg1 == 'interface':
        command = command + bgpNeighborArg1 + ' '
    elif bgpNeighborArg1 == 'local-as':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_local_as', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2 + ' '
            if bgpNeighborArg3 is not None and bgpNeighborArg3 == 'no-prepend':
                command = command + bgpNeighborArg3 + ' '
                if bgpNeighborArg4 is not None and bgpNeighborArg4 == 'replace-as':
                    command = command + bgpNeighborArg4 + ' '
                    if bgpNeighborArg5 is not None and bgpNeighborArg5 == 'dual-as':
                        command = command + bgpNeighborArg5
                    else:
                        command = command.strip()
                else:
                    command = command.strip()
            else:
                command = command.strip()
        else:
            retVal = 'Error-312'
            return retVal
    elif bgpNeighborArg1 == 'maximum-peers':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_maxpeers', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2
        else:
            retVal = 'Error-311'
            return retVal
    elif bgpNeighborArg1 == 'password':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_password', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2
        else:
            retVal = 'Error-310'
            return retVal
    elif bgpNeighborArg1 == 'remove-private-AS':
        command = command + bgpNeighborArg1
    elif bgpNeighborArg1 == 'timers':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_timers_Keepalive', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2 + ' '
            value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_timers_holdtime', bgpNeighborArg3)
            if value == 'ok':
                command = command + bgpNeighborArg3
            else:
                retVal = 'Error-309'
                return retVal
        else:
            retVal = 'Error-308'
            return retVal
    elif bgpNeighborArg1 == 'transport':
        command = command + bgpNeighborArg1 + ' connection-mode passive '
    elif bgpNeighborArg1 == 'ttl-security':
        command = command + bgpNeighborArg1 + ' hops '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_ttl_hops', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2
        else:
            retVal = 'Error-307'
            return retVal
    elif bgpNeighborArg1 == 'update-source':
        command = command + bgpNeighborArg1 + ' '
        if bgpNeighborArg2 is not None:
            value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_update_options', bgpNeighborArg2)
            if value == 'ok':
                command = command + bgpNeighborArg2 + ' '
                if bgpNeighborArg2 == 'ethernet':
                    value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_update_ethernet', bgpNeighborArg3)
                    if value == 'ok':
                        command = command + bgpNeighborArg3
                    else:
                        retVal = 'Error-304'
                        return retVal
                elif bgpNeighborArg2 == 'loopback':
                    value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_update_loopback', bgpNeighborArg3)
                    if value == 'ok':
                        command = command + bgpNeighborArg3
                    else:
                        retVal = 'Error-305'
                        return retVal
                else:
                    value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_update_vlan', bgpNeighborArg3)
                    if value == 'ok':
                        command = command + bgpNeighborArg3
                    else:
                        retVal = 'Error-306'
                        return retVal
            else:
                command = command + bgpNeighborArg2
        else:
            retVal = 'Error-303'
            return retVal
    elif bgpNeighborArg1 == 'weight':
        command = command + bgpNeighborArg1 + ' '
        value = cnos.checkSanityofVariable(deviceType, 'bgp_neighbor_weight', bgpNeighborArg2)
        if value == 'ok':
            command = command + bgpNeighborArg2
        else:
            retVal = 'Error-302'
            return retVal
    else:
        retVal = 'Error-301'
        return retVal
    inner_cmd = [{'command': command, 'prompt': None, 'answer': None}]
    cmd.extend(inner_cmd)
    retVal = retVal + str(cnos.run_cnos_commands(module, cmd))
    command = 'exit \n'
    return retVal