import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
def split_vm_2_gpu_cards_map(vm_2_gpu_cards_map, requested_gpu_num):
    """
    This function split the `vm, all_gpu_cards` map into array of
    "vm, gpu_cards_with_requested_gpu_num" map. The purpose to split the gpu list is for
    avioding GPU contention when creating multiple VMs on one ESXi host.

    Parameters:
        vm_2_gpu_cards_map: It is `vm, all_gpu_cards` map, and you can get it by call
                          function `get_vm_2_gpu_cards_map`.
        requested_gpu_num: The number of GPU cards is requested by each ray node.

    Returns:
        Array of "vm, gpu_cards_with_requested_gpu_num" map.
        Each element of this array will be used in one ray node.

    Example:
        We have 3 hosts, `host1`, `host2`, and `host3`
        Each host has 1 frozen vm, `frozen-vm-1`, `frozen-vm-2`, and `frozen-vm-3`.
        Dynamic passthrough is enabled.
        pciId: 0000:3b:00.0, customLabel:
        `host1` has 3 GPU cards, with pciId/customLabel:
            `0000:3b:00.0/training-0`,
            `0000:3b:00.1/training-1`,
            `0000:3b:00.2/training-2`
        `host2` has 2 GPU cards, with pciId/customLabel:
            `0000:3b:00.3/training-3`,
            `0000:3b:00.4/training-4`
        `host3` has 1 GPU card, with pciId/customLabel:
            `0000:3b:00.5/training-5`
        And we provision a ray cluster with 3 nodes, each node need 1 GPU card

        In this case,  vm_2_gpu_cards_map is like this:
        {
            'frozen-vm-1': [
                pciId: 0000:3b:00.0, customLabel: training-0,
                pciId: 0000:3b:00.1, customLabel: training-1,
                pciId: 0000:3b:00.2, customLabel: training-2,
            ],
            'frozen-vm-2': [
                pciId: 0000:3b:00.3, customLabel: training-3,
                pciId: 0000:3b:00.4, customLabel: training-4,
            ],
            'frozen-vm-3': [ pciId: 0000:3b:00.5, customLabel: training-5 ],
        }
        requested_gpu_num is 1.

        After call the above with this funtion, it returns this array:
        [
            { 'frozen-vm-1' : [ pciId: 0000:3b:00.0, customLabel: training-0 ] },
            { 'frozen-vm-1' : [ pciId: 0000:3b:00.1, customLabel: training-1 ] },
            { 'frozen-vm-1' : [ pciId: 0000:3b:00.2, customLabel: training-2 ] },
            { 'frozen-vm-2' : [ pciId: 0000:3b:00.3, customLabel: training-3 ] },
            { 'frozen-vm-2' : [ pciId: 0000:3b:00.4, customLabel: training-4 ] },
            { 'frozen-vm-3' : [ pciId: 0000:3b:00.5, customLabel: training-5 ] },
        ]

        Each element of this array could be used in 1 ray node with exactly
        `requested_gpu_num` GPU, no more, no less.
    """
    gpu_cards_map_array = []
    for vm_name in vm_2_gpu_cards_map:
        gpu_cards = vm_2_gpu_cards_map[vm_name]
        i = 0
        j = requested_gpu_num
        while j <= len(gpu_cards):
            gpu_cards_map = {vm_name: gpu_cards[i:j]}
            gpu_cards_map_array.append(gpu_cards_map)
            i = j
            j = i + requested_gpu_num
    return gpu_cards_map_array