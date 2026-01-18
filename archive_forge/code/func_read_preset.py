from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def read_preset(self, id=None):
    """
        The ReadPreset operation gets detailed information about a
        preset.

        :type id: string
        :param id: The identifier of the preset for which you want to get
            detailed information.

        """
    uri = '/2012-09-25/presets/{0}'.format(id)
    return self.make_request('GET', uri, expected_status=200)