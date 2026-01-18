from __future__ import annotations
import string
from queue import Empty
from typing import Any, Dict, Set
import azure.core.exceptions
import azure.servicebus.exceptions
import isodate
from azure.servicebus import (ServiceBusClient, ServiceBusMessage,
from azure.servicebus.management import ServiceBusAdministrationClient
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def entity_name(self, name: str, table: dict[int, int] | None=None) -> str:
    """Format AMQP queue name into a valid ServiceBus queue name."""
    return str(safe_str(name)).translate(table or CHARS_REPLACE_TABLE)