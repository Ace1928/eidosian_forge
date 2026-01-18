import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def get_ml_model(self, ml_model_id, verbose=None):
    """
        Returns an `MLModel` that includes detailed metadata, and data
        source information as well as the current status of the
        `MLModel`.

        `GetMLModel` provides results in normal or verbose format.

        :type ml_model_id: string
        :param ml_model_id: The ID assigned to the `MLModel` at creation.

        :type verbose: boolean
        :param verbose: Specifies whether the `GetMLModel` operation should
            return `Recipe`.
        If true, `Recipe` is returned.

        If false, `Recipe` is not returned.

        """
    params = {'MLModelId': ml_model_id}
    if verbose is not None:
        params['Verbose'] = verbose
    return self.make_request(action='GetMLModel', body=json.dumps(params))