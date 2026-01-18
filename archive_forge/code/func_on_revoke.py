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
def on_revoke(self, consumer, topic_partitions):
    """Rebalance on_revoke callback"""
    self.assignment_dict = dict()
    for t in topic_partitions:
        LOG.debug('Topic %s revoked from partition %d', t.topic, t.partition)