import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def enable_uvloop():
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        return False
    else:
        return True