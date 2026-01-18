from novaclient import api_versions
from novaclient import base
@api_versions.wraps('2.81')
def cache_images(self, aggregate, images):
    """
        Request images be cached on a given aggregate.

        :param aggregate: The aggregate to target
        :param images: A list of image IDs to request caching
        :returns: An instance of novaclient.base.TupleWithMeta
        """
    body = {'cache': [{'id': base.getid(image)} for image in images]}
    resp, body = self.api.client.post('/os-aggregates/%s/images' % base.getid(aggregate), body=body)
    return self.convert_into_with_meta(body, resp)