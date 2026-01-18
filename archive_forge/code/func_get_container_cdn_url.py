from libcloud.common.types import LibcloudError
from libcloud.storage.providers import Provider
from libcloud.storage.drivers.s3 import BaseS3Connection, BaseS3StorageDriver
def get_container_cdn_url(self, *argv):
    raise LibcloudError(NO_CDN_SUPPORT_ERROR, driver=self)