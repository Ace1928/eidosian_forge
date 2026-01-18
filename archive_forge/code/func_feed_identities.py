from __future__ import annotations
import hashlib
import hmac
import json
import logging
import os
import pickle
import pprint
import random
import typing as t
import warnings
from binascii import b2a_hex
from datetime import datetime, timezone
from hmac import compare_digest
import zmq.asyncio
from tornado.ioloop import IOLoop
from traitlets import (
from traitlets.config.configurable import Configurable, LoggingConfigurable
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
from zmq.eventloop.zmqstream import ZMQStream
from ._version import protocol_version
from .adapter import adapt
from .jsonutil import extract_dates, json_clean, json_default, squash_dates
def feed_identities(self, msg_list: list[bytes] | list[zmq.Message], copy: bool=True) -> tuple[list[bytes], list[bytes] | list[zmq.Message]]:
    """Split the identities from the rest of the message.

        Feed until DELIM is reached, then return the prefix as idents and
        remainder as msg_list. This is easily broken by setting an IDENT to DELIM,
        but that would be silly.

        Parameters
        ----------
        msg_list : a list of Message or bytes objects
            The message to be split.
        copy : bool
            flag determining whether the arguments are bytes or Messages

        Returns
        -------
        (idents, msg_list) : two lists
            idents will always be a list of bytes, each of which is a ZMQ
            identity. msg_list will be a list of bytes or zmq.Messages of the
            form [HMAC,p_header,p_parent,p_content,buffer1,buffer2,...] and
            should be unpackable/unserializable via self.deserialize at this
            point.
        """
    if copy:
        msg_list = t.cast(t.List[bytes], msg_list)
        idx = msg_list.index(DELIM)
        return (msg_list[:idx], msg_list[idx + 1:])
    else:
        msg_list = t.cast(t.List[zmq.Message], msg_list)
        failed = True
        for idx, m in enumerate(msg_list):
            if m.bytes == DELIM:
                failed = False
                break
        if failed:
            msg = 'DELIM not in msg_list'
            raise ValueError(msg)
        idents, msg_list = (msg_list[:idx], msg_list[idx + 1:])
        return ([bytes(m.bytes) for m in idents], msg_list)