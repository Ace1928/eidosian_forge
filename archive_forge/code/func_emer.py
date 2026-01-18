import datetime
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import ovs.dirs
import ovs.unixctl
import ovs.util
def emer(self, message, **kwargs):
    self.__log('EMER', message, **kwargs)