import abc
import logging
from oslo_messaging.target import Target
def multicast_address(self, target, service=SERVICE_RPC):
    if service == SERVICE_RPC:
        prefix = self._rpc_multicast
    else:
        prefix = self._notify_multicast
    return self._concat('/', [prefix, self._vhost, target.exchange or self._exchange[service], target.topic])