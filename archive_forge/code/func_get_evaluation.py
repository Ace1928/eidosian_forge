import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def get_evaluation(self, evaluation_id):
    """
        Returns an `Evaluation` that includes metadata as well as the
        current status of the `Evaluation`.

        :type evaluation_id: string
        :param evaluation_id: The ID of the `Evaluation` to retrieve. The
            evaluation of each `MLModel` is recorded and cataloged. The ID
            provides the means to access the information.

        """
    params = {'EvaluationId': evaluation_id}
    return self.make_request(action='GetEvaluation', body=json.dumps(params))