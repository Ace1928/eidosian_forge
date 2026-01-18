import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def delete_batch_prediction(self, batch_prediction_id):
    """
        Assigns the DELETED status to a `BatchPrediction`, rendering
        it unusable.

        After using the `DeleteBatchPrediction` operation, you can use
        the GetBatchPrediction operation to verify that the status of
        the `BatchPrediction` changed to DELETED.

        The result of the `DeleteBatchPrediction` operation is
        irreversible.

        :type batch_prediction_id: string
        :param batch_prediction_id: A user-supplied ID that uniquely identifies
            the `BatchPrediction`.

        """
    params = {'BatchPredictionId': batch_prediction_id}
    return self.make_request(action='DeleteBatchPrediction', body=json.dumps(params))