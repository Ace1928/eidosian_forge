import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def get_batch_prediction(self, batch_prediction_id):
    """
        Returns a `BatchPrediction` that includes detailed metadata,
        status, and data file information for a `Batch Prediction`
        request.

        :type batch_prediction_id: string
        :param batch_prediction_id: An ID assigned to the `BatchPrediction` at
            creation.

        """
    params = {'BatchPredictionId': batch_prediction_id}
    return self.make_request(action='GetBatchPrediction', body=json.dumps(params))