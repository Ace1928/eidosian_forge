import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
def set_gpu_placeholder(array_obj, place_holder_number):
    for i in range(place_holder_number):
        array_obj.append({})