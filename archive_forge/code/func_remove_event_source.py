import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def remove_event_source(self, uuid):
    """
        Removes an event source mapping. This means AWS Lambda will no
        longer invoke the function for events in the associated
        source.

        This operation requires permission for the
        `lambda:RemoveEventSource` action.

        :type uuid: string
        :param uuid: The event source mapping ID.

        """
    uri = '/2014-11-13/event-source-mappings/{0}'.format(uuid)
    return self.make_request('DELETE', uri, expected_status=204)