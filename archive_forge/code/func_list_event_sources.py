import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def list_event_sources(self, event_source_arn=None, function_name=None, marker=None, max_items=None):
    """
        Returns a list of event source mappings. For each mapping, the
        API returns configuration information (see AddEventSource).
        You can optionally specify filters to retrieve specific event
        source mappings.

        This operation requires permission for the
        `lambda:ListEventSources` action.

        :type event_source_arn: string
        :param event_source_arn: The Amazon Resource Name (ARN) of the Amazon
            Kinesis stream.

        :type function_name: string
        :param function_name: The name of the AWS Lambda function.

        :type marker: string
        :param marker: Optional string. An opaque pagination token returned
            from a previous `ListEventSources` operation. If present, specifies
            to continue the list from where the returning call left off.

        :type max_items: integer
        :param max_items: Optional integer. Specifies the maximum number of
            event sources to return in response. This value must be greater
            than 0.

        """
    uri = '/2014-11-13/event-source-mappings/'
    params = {}
    headers = {}
    query_params = {}
    if event_source_arn is not None:
        query_params['EventSource'] = event_source_arn
    if function_name is not None:
        query_params['FunctionName'] = function_name
    if marker is not None:
        query_params['Marker'] = marker
    if max_items is not None:
        query_params['MaxItems'] = max_items
    return self.make_request('GET', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)