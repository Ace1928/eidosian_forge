import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def get_and_fixup_unknown_message_content(msg):
    return bytes(msg.get_payload(0))