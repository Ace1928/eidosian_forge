from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import time
from paste.exceptions import formatter
def assemble_email(self, exc_data):
    short_html_version = self.format_html(exc_data, show_hidden_frames=False)
    long_html_version = self.format_html(exc_data, show_hidden_frames=True)
    text_version = self.format_text(exc_data, show_hidden_frames=False)
    msg = MIMEMultipart()
    msg.set_type('multipart/alternative')
    msg.preamble = msg.epilogue = ''
    text_msg = MIMEText(text_version)
    text_msg.set_type('text/plain')
    text_msg.set_param('charset', 'ASCII')
    msg.attach(text_msg)
    html_msg = MIMEText(short_html_version)
    html_msg.set_type('text/html')
    html_msg.set_param('charset', 'UTF-8')
    html_long = MIMEText(long_html_version)
    html_long.set_type('text/html')
    html_long.set_param('charset', 'UTF-8')
    msg.attach(html_msg)
    msg.attach(html_long)
    subject = '%s: %s' % (exc_data.exception_type, formatter.truncate(str(exc_data.exception_value)))
    msg['Subject'] = self.subject_prefix + subject
    msg['From'] = self.from_address
    msg['To'] = ', '.join(self.to_addresses)
    return msg