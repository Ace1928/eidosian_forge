import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def get_data_source(self, data_source_id, verbose=None):
    """
        Returns a `DataSource` that includes metadata and data file
        information, as well as the current status of the
        `DataSource`.

        `GetDataSource` provides results in normal or verbose format.
        The verbose format adds the schema description and the list of
        files pointed to by the DataSource to the normal format.

        :type data_source_id: string
        :param data_source_id: The ID assigned to the `DataSource` at creation.

        :type verbose: boolean
        :param verbose: Specifies whether the `GetDataSource` operation should
            return `DataSourceSchema`.
        If true, `DataSourceSchema` is returned.

        If false, `DataSourceSchema` is not returned.

        """
    params = {'DataSourceId': data_source_id}
    if verbose is not None:
        params['Verbose'] = verbose
    return self.make_request(action='GetDataSource', body=json.dumps(params))