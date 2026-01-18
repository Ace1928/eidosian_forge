import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def query_objects(self, pipeline_id, sphere, marker=None, query=None, limit=None):
    """
        Queries a pipeline for the names of objects that match a
        specified set of conditions.

        The objects returned by QueryObjects are paginated and then
        filtered by the value you set for query. This means the action
        may return an empty result set with a value set for marker. If
        `HasMoreResults` is set to `True`, you should continue to call
        QueryObjects, passing in the returned value for marker, until
        `HasMoreResults` returns `False`.

        :type pipeline_id: string
        :param pipeline_id: Identifier of the pipeline to be queried for object
            names.

        :type query: dict
        :param query: Query that defines the objects to be returned. The Query
            object can contain a maximum of ten selectors. The conditions in
            the query are limited to top-level String fields in the object.
            These filters can be applied to components, instances, and
            attempts.

        :type sphere: string
        :param sphere: Specifies whether the query applies to components or
            instances. Allowable values: `COMPONENT`, `INSTANCE`, `ATTEMPT`.

        :type marker: string
        :param marker: The starting point for the results to be returned. The
            first time you call QueryObjects, this value should be empty. As
            long as the action returns `HasMoreResults` as `True`, you can call
            QueryObjects again and pass the marker value from the response to
            retrieve the next set of results.

        :type limit: integer
        :param limit: Specifies the maximum number of object names that
            QueryObjects will return in a single call. The default value is
            100.

        """
    params = {'pipelineId': pipeline_id, 'sphere': sphere}
    if query is not None:
        params['query'] = query
    if marker is not None:
        params['marker'] = marker
    if limit is not None:
        params['limit'] = limit
    return self.make_request(action='QueryObjects', body=json.dumps(params))