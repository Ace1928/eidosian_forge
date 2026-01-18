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
def get_matched_tags(self, tag_filters, dynamic_id):
    """
        tag_filters will be a dict like {"tag_key1": "val1", "tag_key2": "val2"}
        dynamic_id will be the vSphere object id
        This function will list all the attached tags of the vSphere object, convert
        the string formatted tag to k,v formatted. Then compare the attached tags to
        the ones in the filters.
        Return all the matched tags and all the tags the vSphere object has.
        vsphere_tag_to_kv_pair will ignore the tags not convertable to k,v pairs.
        """
    matched_tags = {}
    all_tags = {}
    for tag_id in self.list_vm_tags(dynamic_id):
        vsphere_vm_tag = self.get_vsphere_sdk_client().tagging.Tag.get(tag_id=tag_id).name
        tag_key_value = vsphere_tag_to_kv_pair(vsphere_vm_tag)
        if tag_key_value:
            tag_key, tag_value = (tag_key_value[0], tag_key_value[1])
            if tag_key in tag_filters and tag_value == tag_filters[tag_key]:
                matched_tags[tag_key] = tag_value
            all_tags[tag_key] = tag_value
    return (matched_tags, all_tags)