import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_connect_network(self, vm, network_name):
    spec = vim.vm.ConfigSpec()
    dev_changes = []
    network_spec = vim.vm.device.VirtualDeviceSpec()
    network_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
    network_spec.device = vim.vm.device.VirtualVmxnet3()
    network_spec.device.backing = vim.vm.device.VirtualEthernetCard.NetworkBackingInfo()
    network_spec.device.backing.useAutoDetect = False
    network_spec.device.backing.network = self.get_obj([vim.Network], network_name)
    network_spec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
    network_spec.device.connectable.startConnected = True
    network_spec.device.connectable.connected = True
    network_spec.device.connectable.allowGuestControl = True
    dev_changes.append(network_spec)
    spec.deviceChange = dev_changes
    output = vm.ReconfigVM_Task(spec=spec)
    print(output.info)