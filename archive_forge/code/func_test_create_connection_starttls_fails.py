import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
def test_create_connection_starttls_fails(self):
    factory = StubSMTPFactory(fail_on=['starttls'], smtp_features=['starttls'])
    conn = self.get_connection(b'', smtp_factory=factory)
    self.assertRaises(smtp_connection.SMTPError, conn._create_connection)
    self.assertEqual([('connect', 'localhost'), ('ehlo',), ('has_extn', 'starttls'), ('starttls',)], factory._calls)