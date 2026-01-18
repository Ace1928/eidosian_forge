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
def find_assignment(self, topic, partition):
    """Find and return existing assignment based on topic and partition"""
    skey = '%s %d' % (topic, partition)
    return self.assignment_dict.get(skey)