from kombu.common import QoS, ignore_errors
from celery import bootsteps
from celery.utils.log import get_logger
from .mingle import Mingle
def set_prefetch_count(prefetch_count):
    return c.task_consumer.qos(prefetch_count=prefetch_count, apply_global=qos_global)