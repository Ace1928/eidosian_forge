from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def read_pipeline(self, id=None):
    """
        The ReadPipeline operation gets detailed information about a
        pipeline.

        :type id: string
        :param id: The identifier of the pipeline to read.

        """
    uri = '/2012-09-25/pipelines/{0}'.format(id)
    return self.make_request('GET', uri, expected_status=200)