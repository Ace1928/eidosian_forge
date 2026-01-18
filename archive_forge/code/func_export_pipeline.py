import deap
from copy import copy
def export_pipeline(exported_pipeline, operators, pset, impute=False, pipeline_score=None, random_state=None, data_file_path=''):
    """Generate source code for a TPOT Pipeline.

    Parameters
    ----------
    exported_pipeline: deap.creator.Individual
        The pipeline that is being exported
    operators:
        List of operator classes from operator library
    pipeline_score:
        Optional pipeline score to be saved to the exported file
    impute: bool (False):
        If impute = True, then adda a imputation step.
    random_state: integer
        Random seed in train_test_split function and exported pipeline.
    data_file_path: string (default: '')
        By default, the path of input dataset is 'PATH/TO/DATA/FILE' by default.
        If data_file_path is another string, the path will be replaced.

    Returns
    -------
    pipeline_text: str
        The source code representing the pipeline

    """
    pipeline_tree = expr_to_tree(exported_pipeline, pset)
    pipeline_text = generate_import_code(exported_pipeline, operators, impute, random_state)
    pipeline_code = pipeline_code_wrapper(generate_export_pipeline_code(pipeline_tree, operators), random_state)
    if pipeline_code.count('FunctionTransformer(copy)'):
        pipeline_text += 'from sklearn.preprocessing import FunctionTransformer\nfrom copy import copy\n'
    if not data_file_path:
        data_file_path = 'PATH/TO/DATA/FILE'
    pipeline_text += "\n# NOTE: Make sure that the outcome column is labeled 'target' in the data file\ntpot_data = pd.read_csv('{}', sep='COLUMN_SEPARATOR', dtype=np.float64)\nfeatures = tpot_data.drop('target', axis=1)\ntraining_features, testing_features, training_target, testing_target = \\\n            train_test_split(features, tpot_data['target'], random_state={})\n".format(data_file_path, random_state)
    if impute:
        pipeline_text += '\nimputer = SimpleImputer(strategy="median")\nimputer.fit(training_features)\ntraining_features = imputer.transform(training_features)\ntesting_features = imputer.transform(testing_features)\n'
    if pipeline_score is not None:
        pipeline_text += '\n# Average CV score on the training set was: {}'.format(pipeline_score)
    pipeline_text += '\n'
    pipeline_text += pipeline_code
    return pipeline_text