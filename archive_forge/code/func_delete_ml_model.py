import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def delete_ml_model(self, ml_model_id):
    """
        Assigns the DELETED status to an `MLModel`, rendering it
        unusable.

        After using the `DeleteMLModel` operation, you can use the
        GetMLModel operation to verify that the status of the
        `MLModel` changed to DELETED.

        The result of the `DeleteMLModel` operation is irreversible.

        :type ml_model_id: string
        :param ml_model_id: A user-supplied ID that uniquely identifies the
            `MLModel`.

        """
    params = {'MLModelId': ml_model_id}
    return self.make_request(action='DeleteMLModel', body=json.dumps(params))