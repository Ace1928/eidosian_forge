from typing import Union
from warnings import warn
from .low_level import *
def new_method_return(parent_msg, signature=None, body=()):
    """Construct a new response message

    :param Message parent_msg: The method call this is a reply to
    :param str signature: The DBus signature of the body data
    :param tuple body: Body data
    """
    header = new_header(MessageType.method_return)
    header.fields[HeaderFields.reply_serial] = parent_msg.header.serial
    sender = parent_msg.header.fields.get(HeaderFields.sender, None)
    if sender is not None:
        header.fields[HeaderFields.destination] = sender
    if signature is not None:
        header.fields[HeaderFields.signature] = signature
    return Message(header, body)