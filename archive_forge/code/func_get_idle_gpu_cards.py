import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
def get_idle_gpu_cards(host, gpu_cards, desired_gpu_number):
    """
    This function takes the number of desired GPU and all the GPU cards of a host.
    This function will select the unused GPU cards and put them into a list.
    If the length of the list > the number of the desired GPU, returns the list,
    otherwise returns an empty list to indicate that this host cannot fulfill the GPU
    requirement.
    """
    gpu_idle_cards = []
    for gpu_card in gpu_cards:
        if is_gpu_available(host, gpu_card):
            gpu_idle_cards.append(gpu_card)
    if len(gpu_idle_cards) < desired_gpu_number:
        logger.warning(f'No enough unused GPU cards on host {host.name}, expected number {desired_gpu_number}, only {len(gpu_idle_cards)}, gpu_cards {gpu_idle_cards}')
        return []
    return gpu_idle_cards