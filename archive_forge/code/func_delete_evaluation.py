import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def delete_evaluation(self, evaluation_id):
    """
        Assigns the `DELETED` status to an `Evaluation`, rendering it
        unusable.

        After invoking the `DeleteEvaluation` operation, you can use
        the GetEvaluation operation to verify that the status of the
        `Evaluation` changed to `DELETED`.

        The results of the `DeleteEvaluation` operation are
        irreversible.

        :type evaluation_id: string
        :param evaluation_id: A user-supplied ID that uniquely identifies the
            `Evaluation` to delete.

        """
    params = {'EvaluationId': evaluation_id}
    return self.make_request(action='DeleteEvaluation', body=json.dumps(params))