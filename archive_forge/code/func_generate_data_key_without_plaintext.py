import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def generate_data_key_without_plaintext(self, key_id, encryption_context=None, key_spec=None, number_of_bytes=None, grant_tokens=None):
    """
        Returns a key wrapped by a customer master key without the
        plaintext copy of that key. To retrieve the plaintext, see
        GenerateDataKey.

        :type key_id: string
        :param key_id: Unique identifier of the key. This can be an ARN, an
            alias, or a globally unique identifier.

        :type encryption_context: map
        :param encryption_context: Name:value pair that contains additional
            data to be authenticated during the encryption and decryption
            processes.

        :type key_spec: string
        :param key_spec: Value that identifies the encryption algorithm and key
            size. Currently this can be AES_128 or AES_256.

        :type number_of_bytes: integer
        :param number_of_bytes: Integer that contains the number of bytes to
            generate. Common values are 128, 256, 512, 1024 and so on.

        :type grant_tokens: list
        :param grant_tokens: A list of grant tokens that represent grants which
            can be used to provide long term permissions to generate a key.

        """
    params = {'KeyId': key_id}
    if encryption_context is not None:
        params['EncryptionContext'] = encryption_context
    if key_spec is not None:
        params['KeySpec'] = key_spec
    if number_of_bytes is not None:
        params['NumberOfBytes'] = number_of_bytes
    if grant_tokens is not None:
        params['GrantTokens'] = grant_tokens
    response = self.make_request(action='GenerateDataKeyWithoutPlaintext', body=json.dumps(params))
    if response.get('CiphertextBlob') is not None:
        response['CiphertextBlob'] = base64.b64decode(response['CiphertextBlob'].encode('utf-8'))
    return response