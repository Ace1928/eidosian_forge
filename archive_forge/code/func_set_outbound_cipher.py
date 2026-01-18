import errno
import os
import socket
import struct
import threading
import time
from hmac import HMAC
from paramiko import util
from paramiko.common import (
from paramiko.util import u
from paramiko.ssh_exception import SSHException, ProxyCommandFailure
from paramiko.message import Message
def set_outbound_cipher(self, block_engine, block_size, mac_engine, mac_size, mac_key, sdctr=False, etm=False):
    """
        Switch outbound data cipher.
        :param etm: Set encrypt-then-mac from OpenSSH
        """
    self.__block_engine_out = block_engine
    self.__sdctr_out = sdctr
    self.__block_size_out = block_size
    self.__mac_engine_out = mac_engine
    self.__mac_size_out = mac_size
    self.__mac_key_out = mac_key
    self.__sent_bytes = 0
    self.__sent_packets = 0
    self.__etm_out = etm
    self.__init_count |= 1
    if self.__init_count == 3:
        self.__init_count = 0
        self.__need_rekey = False