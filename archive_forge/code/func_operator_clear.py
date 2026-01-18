import logging
from os_ken.services.protocols.bgp.api.base import ApiException
from os_ken.services.protocols.bgp.api.base import register
from os_ken.services.protocols.bgp.api.rpc_log_handler import RpcLogHandler
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.commands.clear import ClearCmd
from os_ken.services.protocols.bgp.operator.commands.set import SetCmd
from os_ken.services.protocols.bgp.operator.commands.show import ShowCmd
from os_ken.services.protocols.bgp.operator.internal_api import InternalApi
@register(name='operator.clear')
def operator_clear(**kwargs):
    return operator_run('clear', **kwargs)