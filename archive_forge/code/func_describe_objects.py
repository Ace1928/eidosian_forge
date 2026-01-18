import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def describe_objects(self, object_ids, pipeline_id, marker=None, evaluate_expressions=None):
    """
        Returns the object definitions for a set of objects associated
        with the pipeline. Object definitions are composed of a set of
        fields that define the properties of the object.

        :type pipeline_id: string
        :param pipeline_id: Identifier of the pipeline that contains the object
            definitions.

        :type object_ids: list
        :param object_ids: Identifiers of the pipeline objects that contain the
            definitions to be described. You can pass as many as 25 identifiers
            in a single call to DescribeObjects.

        :type evaluate_expressions: boolean
        :param evaluate_expressions: Indicates whether any expressions in the
            object should be evaluated when the object descriptions are
            returned.

        :type marker: string
        :param marker: The starting point for the results to be returned. The
            first time you call DescribeObjects, this value should be empty. As
            long as the action returns `HasMoreResults` as `True`, you can call
            DescribeObjects again and pass the marker value from the response
            to retrieve the next set of results.

        """
    params = {'pipelineId': pipeline_id, 'objectIds': object_ids}
    if evaluate_expressions is not None:
        params['evaluateExpressions'] = evaluate_expressions
    if marker is not None:
        params['marker'] = marker
    return self.make_request(action='DescribeObjects', body=json.dumps(params))