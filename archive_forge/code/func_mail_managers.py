from django.conf import settings
from django.core.mail.message import (
from django.core.mail.utils import DNS_NAME, CachedDnsName
from django.utils.module_loading import import_string
def mail_managers(subject, message, fail_silently=False, connection=None, html_message=None):
    """Send a message to the managers, as defined by the MANAGERS setting."""
    if not settings.MANAGERS:
        return
    if not all((isinstance(a, (list, tuple)) and len(a) == 2 for a in settings.MANAGERS)):
        raise ValueError('The MANAGERS setting must be a list of 2-tuples.')
    mail = EmailMultiAlternatives('%s%s' % (settings.EMAIL_SUBJECT_PREFIX, subject), message, settings.SERVER_EMAIL, [a[1] for a in settings.MANAGERS], connection=connection)
    if html_message:
        mail.attach_alternative(html_message, 'text/html')
    mail.send(fail_silently=fail_silently)