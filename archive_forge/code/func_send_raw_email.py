import re
import base64
from boto.compat import six, urllib
from boto.connection import AWSAuthConnection
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
import boto
import boto.jsonresponse
from boto.ses import exceptions as ses_exceptions
def send_raw_email(self, raw_message, source=None, destinations=None):
    """Sends an email message, with header and content specified by the
        client. The SendRawEmail action is useful for sending multipart MIME
        emails, with attachments or inline content. The raw text of the message
        must comply with Internet email standards; otherwise, the message
        cannot be sent.

        :type source: string
        :param source: The sender's email address. Amazon's docs say:

          If you specify the Source parameter, then bounce notifications and
          complaints will be sent to this email address. This takes precedence
          over any Return-Path header that you might include in the raw text of
          the message.

        :type raw_message: string
        :param raw_message: The raw text of the message. The client is
          responsible for ensuring the following:

          - Message must contain a header and a body, separated by a blank line.
          - All required header fields must be present.
          - Each part of a multipart MIME message must be formatted properly.
          - MIME content types must be among those supported by Amazon SES.
            Refer to the Amazon SES Developer Guide for more details.
          - Content must be base64-encoded, if MIME requires it.

        :type destinations: list of strings or string
        :param destinations: A list of destinations for the message.

        """
    if isinstance(raw_message, six.text_type):
        raw_message = raw_message.encode('utf-8')
    params = {'RawMessage.Data': base64.b64encode(raw_message)}
    if source:
        params['Source'] = source
    if destinations:
        self._build_list_params(params, destinations, 'Destinations.member')
    return self._make_request('SendRawEmail', params)