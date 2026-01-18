import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
def get_gpu_cards_from_vm(vm, desired_gpu_number, is_dynamic_pci_passthrough):
    """
    This function will be called when there is only one single frozen VM.
    It returns gpu_cards if enough GPUs are available for this VM,
    Or returns an empty list.
    """
    gpu_cards = get_supported_gpus(vm.runtime.host, is_dynamic_pci_passthrough)
    if len(gpu_cards) < desired_gpu_number:
        logger.warning(f'No enough supported GPU cards for VM {vm.name} on host {vm.runtime.host.name}, expected number {desired_gpu_number}, only {len(gpu_cards)}, gpu_cards {gpu_cards}')
        return []
    gpu_idle_cards = get_idle_gpu_cards(vm.runtime.host, gpu_cards, desired_gpu_number)
    if gpu_idle_cards:
        logger.info(f'Got Frozen VM {vm.name}, Host {vm.runtime.host.name}, GPU Cards {gpu_idle_cards}')
    else:
        logger.warning(f'No enough unused GPU cards for VM {vm.name} on host {vm.runtime.host.name}')
    return gpu_idle_cards