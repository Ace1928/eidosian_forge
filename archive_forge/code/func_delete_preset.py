from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def delete_preset(self, id=None):
    """
        The DeletePreset operation removes a preset that you've added
        in an AWS region.

        You can't delete the default presets that are included with
        Elastic Transcoder.

        :type id: string
        :param id: The identifier of the preset for which you want to get
            detailed information.

        """
    uri = '/2012-09-25/presets/{0}'.format(id)
    return self.make_request('DELETE', uri, expected_status=202)