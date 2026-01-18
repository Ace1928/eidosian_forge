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
def target_to_topic(target, priority=None, vhost=None):
    """Convert target into topic string

    :param target: Message destination target
    :type target: oslo_messaging.Target
    :param priority: Notification priority
    :type priority: string
    :param priority: Notification vhost
    :type priority: string
    """
    return concat('.', [target.topic, priority, vhost])