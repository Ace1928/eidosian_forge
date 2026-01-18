import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def prepare_encrypted_request(self, session, endpoint, message):
    """
        Creates a prepared request to send to the server with an encrypted message
        and correct headers

        :param session: The handle of the session to prepare requests with
        :param endpoint: The endpoint/server to prepare requests to
        :param message: The unencrypted message to send to the server
        :return: A prepared request that has an encrypted message
        """
    host = urlsplit(endpoint).hostname
    if self.protocol == 'credssp' and len(message) > self.SIXTEN_KB:
        content_type = 'multipart/x-multi-encrypted'
        encrypted_message = b''
        message_chunks = [message[i:i + self.SIXTEN_KB] for i in range(0, len(message), self.SIXTEN_KB)]
        for message_chunk in message_chunks:
            encrypted_chunk = self._encrypt_message(message_chunk, host)
            encrypted_message += encrypted_chunk
    else:
        content_type = 'multipart/encrypted'
        encrypted_message = self._encrypt_message(message, host)
    encrypted_message += self.MIME_BOUNDARY + b'--\r\n'
    request = requests.Request('POST', endpoint, data=encrypted_message)
    prepared_request = session.prepare_request(request)
    prepared_request.headers['Content-Length'] = str(len(prepared_request.body))
    prepared_request.headers['Content-Type'] = '{0};protocol="{1}";boundary="Encrypted Boundary"'.format(content_type, self.protocol_string.decode())
    return prepared_request