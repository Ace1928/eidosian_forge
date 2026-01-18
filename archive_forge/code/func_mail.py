import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
def mail(self, sender, options=()):
    """SMTP 'mail' command -- begins mail xfer session.

        This method may raise the following exceptions:

         SMTPNotSupportedError  The options parameter includes 'SMTPUTF8'
                                but the SMTPUTF8 extension is not supported by
                                the server.
        """
    optionlist = ''
    if options and self.does_esmtp:
        if any((x.lower() == 'smtputf8' for x in options)):
            if self.has_extn('smtputf8'):
                self.command_encoding = 'utf-8'
            else:
                raise SMTPNotSupportedError('SMTPUTF8 not supported by server')
        optionlist = ' ' + ' '.join(options)
    self.putcmd('mail', 'FROM:%s%s' % (quoteaddr(sender), optionlist))
    return self.getreply()