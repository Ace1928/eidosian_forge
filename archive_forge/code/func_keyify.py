import abc
import logging
from oslo_messaging.target import Target
def keyify(address, service=SERVICE_RPC):
    """Create a hashable key from a Target and service that will uniquely
    identify the generated address. This key is used to map the abstract
    oslo.messaging address to its corresponding AMQP link(s). This mapping may
    be done before the connection is established.
    """
    if isinstance(address, Target):
        return 'Target:{t={%s} e={%s} s={%s} f={%s} service={%s}}' % (address.topic, address.exchange, address.server, address.fanout, service)
    else:
        return 'String:{%s}' % address