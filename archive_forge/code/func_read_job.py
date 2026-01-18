from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def read_job(self, id=None):
    """
        The ReadJob operation returns detailed information about a
        job.

        :type id: string
        :param id: The identifier of the job for which you want to get detailed
            information.

        """
    uri = '/2012-09-25/jobs/{0}'.format(id)
    return self.make_request('GET', uri, expected_status=200)