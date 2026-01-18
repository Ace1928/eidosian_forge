import logging
import threading
import confluent_kafka
from confluent_kafka import KafkaException
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers.kafka_driver import kafka_options
def pack_message(ctxt, msg):
    """Pack context into msg."""
    if isinstance(ctxt, dict):
        context_d = ctxt
    else:
        context_d = ctxt.to_dict()
    msg['_context'] = context_d
    msg = driver_common.serialize_msg(msg)
    return msg