import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_create_image_export_task(self, image: NodeImage=None, osu_export_disk_image_format: str=None, osu_export_api_key_id: str=None, osu_export_api_secret_key: str=None, osu_export_bucket: str=None, osu_export_manifest_url: str=None, osu_export_prefix: str=None, dry_run: bool=False):
    """
        Exports an Outscale machine image (OMI) to an Object Storage Unit
        (OSU) bucket.
        This action enables you to copy an OMI between accounts in different
        Regions. To copy an OMI in the same Region,
        you can also use the CreateImage method.
        The copy of the OMI belongs to you and is
        independent from the source OMI.

        :param      image: The ID of the OMI to export. (required)
        :type       image: ``NodeImage``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :param      osu_export_disk_image_format: The format of the
        export disk (qcow2 | vdi | vmdk). (required)
        :type       osu_export_disk_image_format: ``str``

        :param      osu_export_api_key_id: The API key of the OSU
        account that enables you to access the bucket.
        :type       osu_export_api_key_id: ``str``

        :param      osu_export_api_secret_key: The secret key of the
        OSU account that enables you to access the bucket.
        :type       osu_export_api_secret_key: ``str``

        :param      osu_export_bucket: The name of the OSU bucket
        you want to export the object to. (required)
        :type       osu_export_bucket: ``str``

        :param      osu_export_manifest_url: The URL of the manifest file.
        :type       osu_export_manifest_url: ``str``

        :param      osu_export_prefix: The prefix for the key of
        the OSU object. This key follows this format:
        prefix + object_export_task_id + '.' + disk_image_format.
        :type       osu_export_prefix: ``str``

        :return: the created image export task
        :rtype: ``dict``
        """
    action = 'CreateImageExportTask'
    data = {'DryRun': dry_run, 'OsuExport': {'OsuApiKey': {}}}
    if image is not None:
        data.update({'ImageId': image.id})
    if osu_export_disk_image_format is not None:
        data['OsuExport'].update({'DiskImageFormat': osu_export_disk_image_format})
    if osu_export_bucket is not None:
        data['OsuExport'].update({'OsuBucket': osu_export_bucket})
    if osu_export_manifest_url is not None:
        data['OsuExport'].update({'OsuManifestUrl': osu_export_manifest_url})
    if osu_export_prefix is not None:
        data['OsuExport'].update({'OsuPrefix': osu_export_prefix})
    if osu_export_api_key_id is not None:
        data['OsuExport']['OsuApiKey'].update({'ApiKeyId': osu_export_api_key_id})
    if osu_export_api_secret_key is not None:
        data['OsuExport']['OsuApiKey'].update({'SecretKey': osu_export_api_secret_key})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ImageExportTask']
    return response.json()