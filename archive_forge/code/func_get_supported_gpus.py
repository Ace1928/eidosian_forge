import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
def get_supported_gpus(host, is_dynamic_pci_passthrough):
    """
    This function returns all the supported GPUs on this host,
    currently "supported" means Nvidia GPU.
    """
    gpu_cards = []
    if host.config.graphicsInfo is None:
        return gpu_cards
    for graphics_info in host.config.graphicsInfo:
        if 'nvidia' in graphics_info.vendorName.lower():
            if is_dynamic_pci_passthrough and host.config.assignableHardwareConfig.attributeOverride:
                for attr in host.config.assignableHardwareConfig.attributeOverride:
                    if graphics_info.pciId in attr.instanceId:
                        gpu_card = GPUCard(graphics_info.pciId, attr.value)
                        gpu_cards.append(gpu_card)
                        break
            else:
                gpu_card = GPUCard(graphics_info.pciId)
                gpu_cards.append(gpu_card)
    return gpu_cards