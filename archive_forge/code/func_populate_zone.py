import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def populate_zone(zone):
    zone['timeout'] = zone.get('timeout', defaults.TIMEOUT)
    zone['strict_testing'] = zone.get('strict_testing', defaults.STRICT_TESTING)
    zone['require_sni'] = zone.get('require_sni', defaults.REQUIRE_SNI)
    return zone