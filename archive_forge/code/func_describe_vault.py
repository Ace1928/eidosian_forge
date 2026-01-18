import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def describe_vault(self, vault_name):
    """
        This operation returns information about a vault, including
        the vault's Amazon Resource Name (ARN), the date the vault was
        created, the number of archives it contains, and the total
        size of all the archives in the vault. The number of archives
        and their total size are as of the last inventory generation.
        This means that if you add or remove an archive from a vault,
        and then immediately use Describe Vault, the change in
        contents will not be immediately reflected. If you want to
        retrieve the latest inventory of the vault, use InitiateJob.
        Amazon Glacier generates vault inventories approximately
        daily. For more information, see `Downloading a Vault
        Inventory in Amazon Glacier`_.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Retrieving Vault Metadata in Amazon Glacier`_ and `Describe
        Vault `_ in the Amazon Glacier Developer Guide .

        :type vault_name: string
        :param vault_name: The name of the vault.
        """
    uri = 'vaults/%s' % vault_name
    return self.make_request('GET', uri)