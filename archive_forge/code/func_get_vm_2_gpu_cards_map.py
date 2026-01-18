import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
def get_vm_2_gpu_cards_map(pyvmomi_sdk_provider, pool_name, desired_gpu_number, is_dynamic_pci_passthrough):
    """
    This function returns "vm, gpu_cards" map, the key represents the VM
    and the value lists represents the available GPUs this VM can bind.
    With this map, we can find which frozen VM we can do instant clone to create the
    Ray nodes.
    """
    result = {}
    pool = pyvmomi_sdk_provider.get_pyvmomi_obj([vim.ResourcePool], pool_name)
    if not pool.vm:
        logger.error(f'No frozen-vm in pool {pool.name}')
        return result
    for vm in pool.vm:
        host = vm.runtime.host
        gpu_cards = get_supported_gpus(host, is_dynamic_pci_passthrough)
        if len(gpu_cards) < desired_gpu_number:
            logger.warning(f'No enough supported GPU cards on host {host.name}, expected number {desired_gpu_number}, only {len(gpu_cards)}, gpu_cards {gpu_cards}')
            continue
        gpu_idle_cards = get_idle_gpu_cards(host, gpu_cards, desired_gpu_number)
        if gpu_idle_cards:
            logger.info(f'Got Frozen VM {vm.name}, Host {host.name}, GPU Cards {gpu_idle_cards}')
            result[vm.name] = gpu_idle_cards
    if not result:
        logger.error(f'No enough unused GPU cards for any VMs of pool {pool.name}')
    return result