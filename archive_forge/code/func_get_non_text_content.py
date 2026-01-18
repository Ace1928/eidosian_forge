import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def get_non_text_content(msg):
    return msg.get_payload(decode=True)