import requests
import re
import struct
import sys
from winrm.exceptions import WinRMError
def parse_encrypted_response(self, response):
    """
        Takes in the encrypted response from the server and decrypts it

        :param response: The response that needs to be decrypted
        :return: The unencrypted message from the server
        """
    content_type = response.headers['Content-Type']
    if 'protocol="{0}"'.format(self.protocol_string.decode()) in content_type:
        host = urlsplit(response.request.url).hostname
        msg = self._decrypt_response(response, host)
    else:
        msg = response.text
    return msg