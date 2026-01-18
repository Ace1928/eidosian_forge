import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def re_encrypt(self, ciphertext_blob, destination_key_id, source_encryption_context=None, destination_encryption_context=None, grant_tokens=None):
    """
        Encrypts data on the server side with a new customer master
        key without exposing the plaintext of the data on the client
        side. The data is first decrypted and then encrypted. This
        operation can also be used to change the encryption context of
        a ciphertext.

        :type ciphertext_blob: blob
        :param ciphertext_blob: Ciphertext of the data to re-encrypt.

        :type source_encryption_context: map
        :param source_encryption_context: Encryption context used to encrypt
            and decrypt the data specified in the `CiphertextBlob` parameter.

        :type destination_key_id: string
        :param destination_key_id: Key identifier of the key used to re-encrypt
            the data.

        :type destination_encryption_context: map
        :param destination_encryption_context: Encryption context to be used
            when the data is re-encrypted.

        :type grant_tokens: list
        :param grant_tokens: Grant tokens that identify the grants that have
            permissions for the encryption and decryption process.

        """
    if not isinstance(ciphertext_blob, six.binary_type):
        raise TypeError('Value of argument ``ciphertext_blob`` must be of type %s.' % six.binary_type)
    ciphertext_blob = base64.b64encode(ciphertext_blob)
    params = {'CiphertextBlob': ciphertext_blob, 'DestinationKeyId': destination_key_id}
    if source_encryption_context is not None:
        params['SourceEncryptionContext'] = source_encryption_context
    if destination_encryption_context is not None:
        params['DestinationEncryptionContext'] = destination_encryption_context
    if grant_tokens is not None:
        params['GrantTokens'] = grant_tokens
    response = self.make_request(action='ReEncrypt', body=json.dumps(params))
    if response.get('CiphertextBlob') is not None:
        response['CiphertextBlob'] = base64.b64decode(response['CiphertextBlob'].encode('utf-8'))
    return response