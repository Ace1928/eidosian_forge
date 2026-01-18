import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def create_evaluation(self, evaluation_id, ml_model_id, evaluation_data_source_id, evaluation_name=None):
    """
        Creates a new `Evaluation` of an `MLModel`. An `MLModel` is
        evaluated on a set of observations associated to a
        `DataSource`. Like a `DataSource` for an `MLModel`, the
        `DataSource` for an `Evaluation` contains values for the
        Target Variable. The `Evaluation` compares the predicted
        result for each observation to the actual outcome and provides
        a summary so that you know how effective the `MLModel`
        functions on the test data. Evaluation generates a relevant
        performance metric such as BinaryAUC, RegressionRMSE or
        MulticlassAvgFScore based on the corresponding `MLModelType`:
        `BINARY`, `REGRESSION` or `MULTICLASS`.

        `CreateEvaluation` is an asynchronous operation. In response
        to `CreateEvaluation`, Amazon Machine Learning (Amazon ML)
        immediately returns and sets the evaluation status to
        `PENDING`. After the `Evaluation` is created and ready for
        use, Amazon ML sets the status to `COMPLETED`.

        You can use the GetEvaluation operation to check progress of
        the evaluation during the creation operation.

        :type evaluation_id: string
        :param evaluation_id: A user-supplied ID that uniquely identifies the
            `Evaluation`.

        :type evaluation_name: string
        :param evaluation_name: A user-supplied name or description of the
            `Evaluation`.

        :type ml_model_id: string
        :param ml_model_id: The ID of the `MLModel` to evaluate.
        The schema used in creating the `MLModel` must match the schema of the
            `DataSource` used in the `Evaluation`.

        :type evaluation_data_source_id: string
        :param evaluation_data_source_id: The ID of the `DataSource` for the
            evaluation. The schema of the `DataSource` must match the schema
            used to create the `MLModel`.

        """
    params = {'EvaluationId': evaluation_id, 'MLModelId': ml_model_id, 'EvaluationDataSourceId': evaluation_data_source_id}
    if evaluation_name is not None:
        params['EvaluationName'] = evaluation_name
    return self.make_request(action='CreateEvaluation', body=json.dumps(params))