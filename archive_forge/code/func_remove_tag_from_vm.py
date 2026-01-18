import copy
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any, Dict
import com.vmware.vapi.std.errors_client as ErrorClients
from com.vmware.cis.tagging_client import CategoryModel
from com.vmware.content.library_client import Item
from com.vmware.vapi.std_client import DynamicID
from com.vmware.vcenter.ovf_client import DiskProvisioningType, LibraryItem
from com.vmware.vcenter.vm.hardware_client import Cpu, Memory
from com.vmware.vcenter.vm_client import Power as HardPower
from com.vmware.vcenter_client import VM, Host, ResourcePool
from pyVim.task import WaitForTask
from pyVmomi import vim
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.vsphere.config import (
from ray.autoscaler._private.vsphere.gpu_utils import (
from ray.autoscaler._private.vsphere.pyvmomi_sdk_provider import PyvmomiSdkProvider
from ray.autoscaler._private.vsphere.scheduler import SchedulerFactory
from ray.autoscaler._private.vsphere.utils import Constants, is_ipv4, now_ts
from ray.autoscaler._private.vsphere.vsphere_sdk_provider import VsphereSdkProvider
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
def remove_tag_from_vm(self, tag_key_to_remove, vm_id):
    dynamic_id = DynamicID(type=Constants.TYPE_OF_RESOURCE, id=vm_id)
    for tag_id in self.list_vm_tags(dynamic_id):
        vsphere_vm_tag = self.get_vsphere_sdk_client().tagging.Tag.get(tag_id=tag_id).name
        tag_key_value = vsphere_tag_to_kv_pair(vsphere_vm_tag)
        tag_key = tag_key_value[0] if tag_key_value else None
        if tag_key == tag_key_to_remove:
            logger.debug('Removing tag {} from the VM {}'.format(tag_key, vm_id))
            self.get_vsphere_sdk_client().tagging.TagAssociation.detach(tag_id, dynamic_id)
            break