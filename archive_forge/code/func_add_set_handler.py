import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def add_set_handler(self, typekey, handler):
    self.set_handlers[typekey] = handler