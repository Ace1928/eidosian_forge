import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def parse_mta_sts_record(rec):
    return dict((field.partition('=')[0::2] for field in (field.strip() for field in rec.split(';')) if field))