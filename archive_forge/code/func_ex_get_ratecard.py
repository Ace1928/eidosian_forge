import os
import time
import base64
import binascii
from libcloud.utils import iso8601
from libcloud.utils.py3 import parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.storage.types import ObjectDoesNotExistError
from libcloud.common.azure_arm import AzureResourceManagementConnection
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import Provider
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
def ex_get_ratecard(self, offer_durable_id, currency='USD', locale='en-US', region='US'):
    """
        Get rate card

        :param offer_durable_id: ID of the offer applicable for this
        user account. (e.g. "0026P")
        See http://azure.microsoft.com/en-us/support/legal/offer-details/
        :type offer_durable_id: str

        :param currency: Desired currency for the response (default: "USD")
        :type currency: ``str``

        :param locale: Locale (default: "en-US")
        :type locale: ``str``

        :param region: Region (two-letter code) (default: "US")
        :type region: ``str``

        :return: A dictionary of rates whose ID's correspond to nothing at all
        :rtype: ``dict``
        """
    action = '/subscriptions/%s/providers/Microsoft.Commerce/RateCard' % (self.subscription_id,)
    params = {'api-version': RATECARD_API_VERSION, '$filter': "OfferDurableId eq 'MS-AZR-%s' and Currency eq '%s' and Locale eq '%s' and RegionInfo eq '%s'" % (offer_durable_id, currency, locale, region)}
    r = self.connection.request(action, params=params)
    return r.object