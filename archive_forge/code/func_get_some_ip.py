from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def get_some_ip(host):
    return socket.getaddrinfo(host, None)[0][4][0]