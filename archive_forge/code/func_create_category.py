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
def create_category(self):
    cli_logger.info(f'Creating {Constants.NODE_CATEGORY} category')
    category_spec = self.get_vsphere_sdk_client().tagging.Category.CreateSpec(name=Constants.NODE_CATEGORY, description='Identifies Ray head node and worker nodes', cardinality=CategoryModel.Cardinality.MULTIPLE, associable_types=set())
    category_id = None
    try:
        category_id = self.get_vsphere_sdk_client().tagging.Category.create(category_spec)
    except ErrorClients.Unauthorized as e:
        cli_logger.abort(f'Unauthorized to create the category. Exception: {e}')
    except Exception as e:
        cli_logger.abort(e)
    cli_logger.info(f'Category {category_id} created')
    return category_id